from pomme.utils import get_molar_mass


def test_get_molar_mass():
    """
    Test the molar mass function.
    """
    assert          get_molar_mass('H') == 1.00794
    assert 18.015 < get_molar_mass('H2O')                    < 18.016
    assert 98.07  < get_molar_mass('H2SO4')                  < 98.08
    assert 386    < get_molar_mass('CF3OCF(CF3)CF2OCF2OCF3') < 386.1
    assert 159.6  < get_molar_mass('Fe2O3')                  < 159.7