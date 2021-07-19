from lib.country import Country
from lib.coalition import Coalition


country_A = Country(name="A", base_temp=20., delta_temp=1., ideal_temp=13.,
                    m_damage=1., power=0.25)

country_B = Country(name="B", base_temp=10., delta_temp=3., ideal_temp=13.,
                    m_damage=1., power=0.25)

coalition = Coalition([country_A, country_B])


def test_total_power():
    assert coalition.total_power == 0.5


def test_avg_ideal_G():
    assert coalition.avg_ideal_G == 4.
