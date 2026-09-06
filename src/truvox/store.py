"""The verified store: the one piece of state that makes corrections
compound instead of evaporating. Once a human confirms a pronunciation,
it outranks every model call for that (word, locale[, identity]) forever —
that's the whole point of "reuse high-confidence results" instead of
re-guessing the same name every time a voice agent sees it again.

SQLite, not a bare JSON file, mainly so `identity_id` can be optional
without hand-rolling merge logic: a NULL/empty identity_id row is the
"generic" pronunciation for a word+locale, and a specific identity_id row
(e.g. a particular patient or student) overrides it. `get()` checks the
specific row first and falls back to the generic one.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from truvox.schema import VerifiedEntry

_GENERIC_IDENTITY = ""  # sentinel for "no specific identity" — see module docstring

_SCHEMA = """
CREATE TABLE IF NOT EXISTS verified_entries (
    word          TEXT NOT NULL,
    locale        TEXT NOT NULL,
    identity_id   TEXT NOT NULL DEFAULT '',
    ipa           TEXT NOT NULL,
    corrected_by  TEXT NOT NULL,
    corrected_at  TEXT NOT NULL,
    notes         TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (word, locale, identity_id)
);
"""


def _norm(value: str) -> str:
    return value.strip().lower()


class VerifiedStore:
    def __init__(self, db_path: str | Path = "data/verified_store.db"):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(_SCHEMA)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self._db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def correct(
        self,
        word: str,
        locale: str,
        ipa: str,
        corrected_by: str,
        identity_id: str | None = None,
        notes: str = "",
    ) -> VerifiedEntry:
        """Record (or overwrite) a human-confirmed pronunciation."""
        if not ipa.strip():
            raise ValueError("ipa must be non-empty")
        entry = VerifiedEntry(
            word=_norm(word),
            locale=_norm(locale),
            ipa=ipa,
            corrected_by=corrected_by,
            identity_id=_norm(identity_id) if identity_id else None,
            notes=notes,
            corrected_at=datetime.now(timezone.utc),
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO verified_entries
                    (word, locale, identity_id, ipa, corrected_by, corrected_at, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (word, locale, identity_id) DO UPDATE SET
                    ipa=excluded.ipa,
                    corrected_by=excluded.corrected_by,
                    corrected_at=excluded.corrected_at,
                    notes=excluded.notes
                """,
                (
                    entry.word,
                    entry.locale,
                    entry.identity_id or _GENERIC_IDENTITY,
                    entry.ipa,
                    entry.corrected_by,
                    entry.corrected_at.isoformat(),
                    entry.notes,
                ),
            )
        return entry

    def get(
        self, word: str, locale: str, identity_id: str | None = None
    ) -> VerifiedEntry | None:
        """Specific-identity entry first, generic entry second, None if
        nothing has ever been verified for this word+locale.
        """
        word, locale = _norm(word), _norm(locale)
        candidates = [_norm(identity_id)] if identity_id else []
        candidates.append(_GENERIC_IDENTITY)

        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            for candidate_identity in candidates:
                row = conn.execute(
                    """
                    SELECT word, locale, identity_id, ipa, corrected_by, corrected_at, notes
                    FROM verified_entries
                    WHERE word = ? AND locale = ? AND identity_id = ?
                    """,
                    (word, locale, candidate_identity),
                ).fetchone()
                if row is not None:
                    return VerifiedEntry(
                        word=row["word"],
                        locale=row["locale"],
                        ipa=row["ipa"],
                        corrected_by=row["corrected_by"],
                        identity_id=row["identity_id"] or None,
                        notes=row["notes"],
                        corrected_at=datetime.fromisoformat(row["corrected_at"]),
                    )
        return None

    def all_entries(self) -> list[VerifiedEntry]:
        """Everything in the store — used by the demo script and tests."""
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT word, locale, identity_id, ipa, corrected_by, corrected_at, notes
                FROM verified_entries ORDER BY corrected_at DESC
                """
            ).fetchall()
        return [
            VerifiedEntry(
                word=r["word"],
                locale=r["locale"],
                ipa=r["ipa"],
                corrected_by=r["corrected_by"],
                identity_id=r["identity_id"] or None,
                notes=r["notes"],
                corrected_at=datetime.fromisoformat(r["corrected_at"]),
            )
            for r in rows
        ]
