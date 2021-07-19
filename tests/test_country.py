from lib.country import Country


country = Country(name="A", base_temp=20., delta_temp=1., ideal_temp=13.,
                  m_damage=1., power=0.25)


def test_climate_change_temp():
    assert country.climate_change_temp == 21.


def test_ideal_geoengineering_level():
    assert country.ideal_geoengineering_level == 8.


def test_climate_change_damage():
    assert country.climate_change_damage == 64.


def test_weighted_damage():
    assert country.weighted_damage == 0.25


def test_damage_without_geo():
    assert country.damage(0.) == 0.


def test_damage_with_geo():
    assert country.damage(6.) == -60.


def test_damage_ideal():
    ideal_G = country.ideal_geoengineering_level
    assert country.damage(ideal_G) == -country.climate_change_damage


def test_payoff():
    assert country.payoff(4.) == 48.
