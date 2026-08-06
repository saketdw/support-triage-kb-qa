"""Smoke test: the packages import. Real tests arrive with each part."""

def test_packages_import():
    import triage
    import kbqa

    assert triage.__doc__ and kbqa.__doc__
