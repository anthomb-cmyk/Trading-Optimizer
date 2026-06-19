"""
tests/test_llm_sentiment.py
===========================
Tests for the cached LLM sentiment scorer (config/llm_sentiment.py).

No network is used:
  - The no-key path is exercised by clearing ANTHROPIC_API_KEY (returns Nones,
    never raises).
  - The Claude path is exercised with a stubbed/mocked client so a numeric
    reply is parsed without any HTTP call.
  - The disk cache is redirected to a temp dir and verified to make re-runs
    free (no second model call).

Runs without pytest:  python3 tests/test_llm_sentiment.py
Runs under pytest too (all public functions are named test_*).

Dependencies: stdlib only.
"""
import sys
import os
import json
import tempfile
import shutil
import importlib

# Allow both `python3 tests/test_llm_sentiment.py` and `pytest` from repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Module loading with an isolated cache dir ───────────────────────────────────
def _fresh_module(cache_root):
    """
    Import config.llm_sentiment fresh with CACHE_DIR pointed at a temp folder
    so tests never touch the real data cache and stay independent.
    """
    import config.llm_sentiment as mod
    mod = importlib.reload(mod)
    mod.CACHE_DIR = os.fspath  # placeholder, replaced below
    import pathlib
    mod.CACHE_DIR = pathlib.Path(cache_root) / "llm_sentiment_cache"
    return mod


class _StubBlock:
    def __init__(self, text):
        self.text = text


class _StubMessage:
    def __init__(self, text):
        self.content = [_StubBlock(text)]


class _StubMessages:
    """Records every call so tests can assert how many times the model ran."""
    def __init__(self, reply_text):
        self._reply_text = reply_text
        self.calls = 0

    def create(self, **_kwargs):
        self.calls += 1
        return _StubMessage(self._reply_text)


class _StubClient:
    def __init__(self, reply_text):
        self.messages = _StubMessages(reply_text)


# ── Tests ───────────────────────────────────────────────────────────────────────
def test_no_key_returns_nones_without_raising():
    """With no API key, every text scores None and nothing is raised."""
    tmp = tempfile.mkdtemp(prefix="llmsent_nokey_")
    try:
        prev = os.environ.pop("ANTHROPIC_API_KEY", None)
        mod = _fresh_module(tmp)
        texts = ["Stocks soar to record high", "Markets crash on rate fears"]
        out = mod.score_texts(texts)
        assert out == [None, None], out
        assert len(out) == len(texts)
        # No cache files should have been written for None results.
        assert not (mod.CACHE_DIR.exists() and any(mod.CACHE_DIR.iterdir()))
        if prev is not None:
            os.environ["ANTHROPIC_API_KEY"] = prev
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("ok: no-key fallback returns Nones without raising")


def test_empty_input():
    """Empty input returns an empty list."""
    tmp = tempfile.mkdtemp(prefix="llmsent_empty_")
    try:
        mod = _fresh_module(tmp)
        assert mod.score_texts([]) == []
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("ok: empty input returns []")


def test_parse_score_variants():
    """The reply parser handles signs, prose, and clamps to [-1, 1]."""
    tmp = tempfile.mkdtemp(prefix="llmsent_parse_")
    try:
        mod = _fresh_module(tmp)
        assert mod._parse_score("0.5") == 0.5
        assert mod._parse_score("+0.75") == 0.75
        assert mod._parse_score("-1") == -1.0
        assert mod._parse_score("The score is 0.2") == 0.2
        assert mod._parse_score("2.0") == 1.0      # clamped high
        assert mod._parse_score("-3") == -1.0      # clamped low
        assert mod._parse_score("no number here") is None
        assert mod._parse_score("") is None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("ok: parser handles numeric replies and clamps")


def test_stubbed_claude_parses_numeric_reply_and_caches(monkeypatch=None):
    """
    With a key present and the anthropic client stubbed, a numeric reply is
    parsed and persisted to the cache; the second run is free (no new call).
    """
    tmp = tempfile.mkdtemp(prefix="llmsent_stub_")
    try:
        os.environ["ANTHROPIC_API_KEY"] = "test-key-not-real"
        mod = _fresh_module(tmp)

        # Stub the anthropic package + client so no HTTP happens.
        import types
        stub_client = _StubClient(reply_text="0.6")
        fake_anthropic = types.ModuleType("anthropic")
        fake_anthropic.Anthropic = lambda *a, **k: stub_client
        sys.modules["anthropic"] = fake_anthropic

        texts = ["Earnings beat expectations, shares jump"]

        # First run hits the (stubbed) model.
        out1 = mod.score_texts(texts)
        assert out1 == [0.6], out1
        assert stub_client.messages.calls == 1, stub_client.messages.calls

        # Cache file should now exist with the score.
        files = list(mod.CACHE_DIR.iterdir())
        assert len(files) == 1, files
        payload = json.loads(files[0].read_text(encoding="utf-8"))
        assert payload["score"] == 0.6, payload

        # Second run is served from cache: no additional model call.
        out2 = mod.score_texts(texts)
        assert out2 == [0.6], out2
        assert stub_client.messages.calls == 1, stub_client.messages.calls
    finally:
        sys.modules.pop("anthropic", None)
        os.environ.pop("ANTHROPIC_API_KEY", None)
        shutil.rmtree(tmp, ignore_errors=True)
    print("ok: stubbed Claude parses numeric reply and cache makes re-runs free")


def test_stubbed_claude_unparseable_reply_returns_none():
    """A non-numeric model reply yields None and is not cached."""
    tmp = tempfile.mkdtemp(prefix="llmsent_bad_")
    try:
        os.environ["ANTHROPIC_API_KEY"] = "test-key-not-real"
        mod = _fresh_module(tmp)

        import types
        stub_client = _StubClient(reply_text="I cannot rate this")
        fake_anthropic = types.ModuleType("anthropic")
        fake_anthropic.Anthropic = lambda *a, **k: stub_client
        sys.modules["anthropic"] = fake_anthropic

        out = mod.score_texts(["Some ambiguous headline"])
        assert out == [None], out
        # Nothing cached for a failed parse.
        assert not (mod.CACHE_DIR.exists() and any(mod.CACHE_DIR.iterdir()))
    finally:
        sys.modules.pop("anthropic", None)
        os.environ.pop("ANTHROPIC_API_KEY", None)
        shutil.rmtree(tmp, ignore_errors=True)
    print("ok: unparseable reply returns None, not cached")


def test_cache_hit_does_not_require_key():
    """A pre-seeded cache entry is returned even with no key (free re-run)."""
    tmp = tempfile.mkdtemp(prefix="llmsent_cachehit_")
    try:
        os.environ.pop("ANTHROPIC_API_KEY", None)
        mod = _fresh_module(tmp)
        text = "Fed signals rate cut, equities rally"

        # Pre-seed the cache directly.
        mod.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        mod._cache_set(text, 0.42)

        out = mod.score_texts([text])
        assert out == [0.42], out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("ok: cached score served without a key")


# ── Standalone runner ───────────────────────────────────────────────────────────
def _run_all():
    tests = [
        test_no_key_returns_nones_without_raising,
        test_empty_input,
        test_parse_score_variants,
        test_stubbed_claude_parses_numeric_reply_and_caches,
        test_stubbed_claude_unparseable_reply_returns_none,
        test_cache_hit_does_not_require_key,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as exc:
            failed += 1
            print(f"FAIL: {t.__name__}: {exc}")
        except Exception as exc:
            failed += 1
            print(f"ERROR: {t.__name__}: {exc!r}")
    if failed:
        print(f"\n{failed} test(s) failed.")
        return 1
    print("\nAll llm_sentiment tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(_run_all())
