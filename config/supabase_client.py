"""Lazy Supabase client singleton. Returns None if env vars are not set."""
from __future__ import annotations

import os
from functools import lru_cache

from config.logger import get_logger

log = get_logger(__name__)


@lru_cache(maxsize=1)
def get_client():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        log.debug("SUPABASE_URL / SUPABASE_KEY not set — Supabase logging disabled.")
        return None
    try:
        from supabase import create_client
        client = create_client(url, key)
        log.info("Supabase client initialised (%s)", url)
        return client
    except Exception as exc:
        log.warning("Could not initialise Supabase client: %s", exc)
        return None
