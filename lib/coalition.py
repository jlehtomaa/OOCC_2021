import numpy as np
from typing import List
from lib.country import Country


class Coalition:
    """Coalition is a collection of cooperating Country instances.

    Arguments:
        members: list (Country) of countries.
    """
    def __init__(self, members: List[Country]):
        self.members = members
        assert all(isinstance(country, Country) for country in self.members)

    @property
    def total_power(self) -> float:
        """Coalition's total global power share.

        Equals the sum of all members' individual powers.
        """
        power = np.sum([country.power for country in self.members])
        assert 0. <= power <= 1., "Coalition's total power must be in [0,1]."
        return power

    @property
    def avg_ideal_G(self) -> float:
        """ Eq. (B.9). Coalition's average ideal geoengineering level."""
        alphas = [country.ideal_geoengineering_level
                  for country in self.members]
        etas = [country.weighted_damage for country in self.members]
        return np.dot(alphas, etas) / np.sum(etas)
