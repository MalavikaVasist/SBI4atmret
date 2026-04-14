import sbi4exoplanets


def test_import_package():
    assert hasattr(sbi4exoplanets, '__name__')
    assert sbi4exoplanets.__name__ == 'sbi4exoplanets'
