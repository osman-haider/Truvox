from truvox.store import VerifiedStore


def test_get_returns_none_when_nothing_stored(tmp_path):
    store = VerifiedStore(tmp_path / "store.db")
    assert store.get("saoirse", "en-IE") is None


def test_correct_then_get_roundtrips(tmp_path):
    store = VerifiedStore(tmp_path / "store.db")
    store.correct("saoirse", "en-IE", "ˈsɪərʃə", corrected_by="usman")

    entry = store.get("Saoirse", "EN-ie")  # case-insensitive lookup
    assert entry is not None
    assert entry.ipa == "ˈsɪərʃə"
    assert entry.corrected_by == "usman"


def test_correcting_twice_overwrites_not_duplicates(tmp_path):
    store = VerifiedStore(tmp_path / "store.db")
    store.correct("saoirse", "en-IE", "wrong-first-guess", corrected_by="usman")
    store.correct("saoirse", "en-IE", "ˈsɪərʃə", corrected_by="usman")

    assert store.get("saoirse", "en-IE").ipa == "ˈsɪərʃə"
    assert len(store.all_entries()) == 1


def test_identity_specific_entry_overrides_generic_entry(tmp_path):
    store = VerifiedStore(tmp_path / "store.db")
    store.correct("kamala", "en-US", "ˈkʌmələ", corrected_by="usman")
    store.correct(
        "kamala", "en-US", "kəˈmɑːlə", corrected_by="usman", identity_id="patient-42"
    )

    assert store.get("kamala", "en-US").ipa == "ˈkʌmələ"
    assert store.get("kamala", "en-US", identity_id="patient-42").ipa == "kəˈmɑːlə"
    # a different, unconfirmed identity still falls back to the generic entry
    assert store.get("kamala", "en-US", identity_id="patient-99").ipa == "ˈkʌmələ"
