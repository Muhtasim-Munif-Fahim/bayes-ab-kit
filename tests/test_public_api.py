import bayes_ab_kit
from bayes_ab_kit import __all__ as public_names


def test_version_is_exposed():
    assert isinstance(bayes_ab_kit.__version__, str)
    assert bayes_ab_kit.__version__.count(".") == 2


def test_all_public_names_are_importable_and_real():
    assert len(public_names) >= 15
    for name in public_names:
        assert getattr(bayes_ab_kit, name, None) is not None, name


def test_no_duplicate_exports():
    assert len(public_names) == len(set(public_names))