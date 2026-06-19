"""
config/llm_sentiment.py
=======================
Cached sentiment scorer for market-news headlines, with an optional Claude
upgrade path and a free fallback.

Public API
----------
    score_texts(texts: list[str]) -> list[float | None]
        Returns a sentiment score in [-1, +1] per input text.

Behaviour
---------
- If ``ANTHROPIC_API_KEY`` is set in the environment AND the ``anthropic``
  package is importable, each text is scored with Claude Haiku
  (``claude-haiku-4-5-20251001``) using a STRICT, wording-only prompt.
- Otherwise every text scores ``None`` so the caller can fall back to the
  free GDELT tone signal.

Costs are kept low by:
  - Disk-caching every score keyed by a SHA-256 hash of the text
    (JSON files under ``DATA_DIR / "llm_sentiment_cache"``). Re-runs are free.
  - Only sending cache-misses to the model, batched into one request.

This module NEVER raises. Any failure (missing dep, API error, bad reply,
unparseable number) is caught, logged, and yields ``None`` for the affected
texts so the trading pipeline degrades gracefully instead of crashing.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import List, Optional

from config.settings import DATA_DIR

try:
    from config.logger import get_logger
    _log = get_logger("llm_sentiment")
except Exception:  # pragma: no cover - logging must never break scoring
    import logging
    _log = logging.getLogger("llm_sentiment")

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL = "claude-haiku-4-5-20251001"
CACHE_DIR = Path(DATA_DIR) / "llm_sentiment_cache"

# Strict, wording-only prompt. Judges tone of the headline, not the outcome.
PROMPT_TEMPLATE = (
    "Rate ONLY the sentiment/tone expressed in this market-news headline from "
    "-1 (very bearish) to +1 (very bullish). Judge the wording, NOT what later "
    "happened. Reply with just the number.\n\n"
    "Headline: {text}"
)

# Keep batches modest so a single bad request can't sink a huge run, and so
# we stay cheap. Each headline is an independent one-shot message.
_BATCH_SIZE = 20

# Matches an optionally-signed decimal anywhere in a short reply, e.g.
# "0.4", "+0.75", "-1", "The score is 0.2".
_NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+")


# ── Cache helpers ───────────────────────────────────────────────────────────────
def _hash_text(text: str) -> str:
    """Stable hash of a text used as the cache key / filename."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cache_path(text: str) -> Path:
    return CACHE_DIR / f"{_hash_text(text)}.json"


def _cache_get(text: str) -> Optional[float]:
    """Return a cached score for ``text`` or ``None`` if not cached/unreadable."""
    path = _cache_path(text)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        score = payload.get("score")
        return float(score) if score is not None else None
    except Exception as exc:  # corrupt cache file - ignore, will re-score
        _log.warning("Ignoring unreadable sentiment cache %s: %s", path.name, exc)
        return None


def _cache_set(text: str, score: float) -> None:
    """Persist ``score`` for ``text`` to disk. Never raises."""
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = _cache_path(text)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"text": text, "score": score}, fh)
    except Exception as exc:  # pragma: no cover - disk issues shouldn't crash
        _log.warning("Failed to write sentiment cache for a headline: %s", exc)


# ── Parsing ─────────────────────────────────────────────────────────────────────
def _parse_score(reply: str) -> Optional[float]:
    """
    Extract a numeric sentiment from a model reply and clamp to [-1, +1].
    Returns ``None`` if no number can be found.
    """
    if not reply:
        return None
    match = _NUMBER_RE.search(reply)
    if not match:
        return None
    try:
        value = float(match.group(0))
    except (TypeError, ValueError):
        return None
    # Clamp defensively - the prompt asks for [-1, 1] but models can drift.
    return max(-1.0, min(1.0, value))


# ── Claude scoring ──────────────────────────────────────────────────────────────
def _claude_available() -> bool:
    """True iff a key is set and the anthropic package is importable."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
    except Exception:
        return False
    return True


def _score_with_claude(texts: List[str]) -> List[Optional[float]]:
    """
    Score each text with Claude Haiku. Returns one score (or None) per text,
    aligned to the input order. Never raises.
    """
    try:
        import anthropic
    except Exception as exc:  # pragma: no cover - guarded by _claude_available
        _log.warning("anthropic import failed at score time: %s", exc)
        return [None] * len(texts)

    try:
        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    except Exception as exc:
        _log.warning("Could not construct Anthropic client: %s", exc)
        return [None] * len(texts)

    scores: List[Optional[float]] = []
    for text in texts:
        score: Optional[float] = None
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=10,
                messages=[
                    {"role": "user", "content": PROMPT_TEMPLATE.format(text=text)}
                ],
            )
            reply = ""
            for block in getattr(resp, "content", []) or []:
                reply += getattr(block, "text", "") or ""
            score = _parse_score(reply)
            if score is None:
                _log.warning("Unparseable sentiment reply %r for headline", reply)
        except Exception as exc:  # API / network / rate-limit etc.
            _log.warning("Claude sentiment scoring failed for a headline: %s", exc)
            score = None
        scores.append(score)
    return scores


# ── Public API ──────────────────────────────────────────────────────────────────
def score_texts(texts: List[str]) -> List[Optional[float]]:
    """
    Score the sentiment/tone of each headline in ``texts``.

    Returns a list the same length as ``texts``, each element a float in
    [-1, +1] (sentiment) or ``None`` when scoring is unavailable/failed.
    Cached scores are reused; only cache-misses hit the model. Never raises.
    """
    try:
        if not texts:
            return []

        # 1) Serve from disk cache where possible.
        results: List[Optional[float]] = [None] * len(texts)
        missing_idx: List[int] = []
        for i, text in enumerate(texts):
            cached = _cache_get(text)
            if cached is not None:
                results[i] = cached
            else:
                missing_idx.append(i)

        if not missing_idx:
            return results

        # 2) Without Claude, leave misses as None for the GDELT fallback.
        if not _claude_available():
            return results

        # 3) Score misses with Claude, batched, and persist successes.
        for start in range(0, len(missing_idx), _BATCH_SIZE):
            batch_idx = missing_idx[start:start + _BATCH_SIZE]
            batch_texts = [texts[i] for i in batch_idx]
            batch_scores = _score_with_claude(batch_texts)
            for idx, score in zip(batch_idx, batch_scores):
                if score is not None:
                    results[idx] = score
                    _cache_set(texts[idx], score)

        return results
    except Exception as exc:  # pragma: no cover - absolute last-resort guard
        _log.warning("score_texts failed unexpectedly: %s", exc)
        return [None] * len(texts)
