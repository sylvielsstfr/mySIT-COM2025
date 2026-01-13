import mysitcom


def test_version():
    """Check to see that we can get the package version"""
    assert mysitcom.__version__ is not None
