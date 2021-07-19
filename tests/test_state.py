from lib.country import Country
from lib.coalition import Coalition
from lib.state import State


country_A = Country(name="A", base_temp=15., delta_temp=5., ideal_temp=10.,
                    m_damage=1., power=0.60)

country_B = Country(name="B", base_temp=10., delta_temp=3., ideal_temp=13.,
                    m_damage=1., power=0.30)

country_C = Country(name="C", base_temp=20., delta_temp=3., ideal_temp=13.,
                    m_damage=1., power=0.10)

all_countries = [country_A, country_B, country_C]


def test_strongest_coalition_weak_governance():
    coalitions = [Coalition([country_A]),
                  Coalition([country_B]),
                  Coalition([country_C])]

    state = State(name="(test_name)",
                  coalitions=coalitions,
                  all_countries=all_countries,
                  power_rule="weak_governance")

    assert state.strongest_coalition == coalitions[0]


def test_strongest_coalition_power_threshold():
    coalitions = [Coalition([country_A]),
                  Coalition([country_B]),
                  Coalition([country_C])]

    state = State(name="(test_name)",
                  coalitions=coalitions,
                  all_countries=all_countries,
                  power_rule="power_threshold",
                  min_power=0.55)

    assert state.strongest_coalition == coalitions[0]


def test_geo_deployment_level_power_threshold():
    coalitions = [Coalition([country_A]),
                  Coalition([country_B]),
                  Coalition([country_C])]

    state = State(name="(test_name)",
                  coalitions=coalitions,
                  all_countries=all_countries,
                  power_rule="power_threshold",
                  min_power=0.55)

    assert state.geo_deployment_level == 10.


def test_geo_deployment_level_power_threshold_not_met():
    coalitions = [Coalition([country_A]),
                  Coalition([country_B]),
                  Coalition([country_C])]

    state = State(name="(test_name)",
                  coalitions=coalitions,
                  all_countries=all_countries,
                  power_rule="power_threshold",
                  min_power=0.75)

    assert state.geo_deployment_level == 0.


def test_payoffs_no_geo():
    coalitions = [Coalition([country_A]),
                  Coalition([country_B]),
                  Coalition([country_C])]

    state = State(name="(test_name)",
                  coalitions=coalitions,
                  all_countries=all_countries,
                  power_rule="power_threshold",
                  min_power=0.75)

    assert state.payoffs == {"A": 0., "B": 0., "C": 0.}


def test_payoffs_G10():
    coalitions = [Coalition([country_A, country_C]),
                  Coalition([country_B])]

    state = State(name="(test_name)",
                  coalitions=coalitions,
                  all_countries=all_countries,
                  power_rule="power_threshold",
                  min_power=0.70)

    assert state.payoffs == {"A": 100., "B": -100., "C": 100.}
