from lib.country import Country
from lib.coalition import Coalition
from lib.state import State
from lib.utils import get_payoff_matrix, get_geoengineering_levels


country_A = Country(name="A", base_temp=10., delta_temp=3., ideal_temp=13.,
                    m_damage=1., power=0.60)

country_B = Country(name="B", base_temp=11., delta_temp=2., ideal_temp=13.,
                    m_damage=1., power=0.30)

country_C = Country(name="C", base_temp=12., delta_temp=1., ideal_temp=13.,
                    m_damage=1., power=0.10)

all_countries = [country_A, country_B, country_C]

state_1 = State(name="state_1",
                coalitions=[Coalition([country_A]),
                            Coalition([country_B]),
                            Coalition([country_C])],
                all_countries=all_countries,
                power_rule="weak_governance")

state_2 = State(name="state_2",
                coalitions=[Coalition([country_A, country_B]),
                            Coalition([country_C])],
                all_countries=all_countries,
                power_rule="weak_governance")

state_3 = State(name="state_3",
                coalitions=[Coalition([country_A, country_B, country_C])],
                all_countries=all_countries,
                power_rule="weak_governance")

states = [state_1, state_2, state_3]


def test_get_payoff_matrix_all_zeros():

    # All countries have ideal geoengineering level at 0,
    # so payoffs are also zeros.
    columns = ["A", "B", "C"]
    df = get_payoff_matrix(states=states, columns=columns)
    assert bool((df == 0.).all().all())


def test_get_geoengineering_levels_all_zeros():
    G = get_geoengineering_levels(states)
    assert all(val == 0. for val in G.values())
