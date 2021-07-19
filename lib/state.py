import numpy as np
from lib.country import Country
from lib.coalition import Coalition


class State:
    """ Encodes the current state of the dynamic coalition game.

    Arguments:
      name: Coalition's name (e.g., '(TC)').
      coalitions: List of Country instances that make up the coalition.
      all_countries: List of all Country instances that exist in the game.
      power_rule: Rule to determine the strongest coalition.
      min_power: Minimum required world power share to do geoengineering.
    """
    def __init__(self,
                 name: str,
                 coalitions: list,
                 all_countries: list,
                 power_rule: str,
                 min_power: float = None):

        # Safety checks.
        assert all(isinstance(country, Country) for country in all_countries)
        assert all(isinstance(coalition, Coalition)
                   for coalition in coalitions)

        self.name = name
        self.coalitions = coalitions
        self.all_countries = all_countries
        self.power_rule = power_rule
        self.min_power = min_power
        self.coalition_powers = [c.total_power for c in self.coalitions]

        assert np.isclose(np.sum(self.coalition_powers), 1., atol=1e-12),\
            "Coalition powers must sum up to 1."

    @property
    def strongest_coalition(self) -> Coalition:
        """
        Returns the coalition that, according to self.power_rule,
        gets to implement geoengineering.

        Note: we assume for the strongest_coalition to be unique.
        """
        if self.power_rule == "power_threshold":
            def sort_key(coalition): return coalition.total_power
        elif self.power_rule == "weak_governance":
            def sort_key(coalition): return coalition.avg_ideal_G
        else:
            msg = ("Incorrect power threshold specification. "
                   "Must be in ['power_threshold', 'weak_governance']")
            raise ValueError(msg)

        sorted_coalitions = sorted(self.coalitions, key=sort_key, reverse=True)
        strongest_coalition = sorted_coalitions[0]
        assert isinstance(strongest_coalition, Coalition)

        return strongest_coalition

    @property
    def geo_deployment_level(self) -> float:
        winner_power = self.strongest_coalition.total_power
        G = self.strongest_coalition.avg_ideal_G

        if self.power_rule == "power_threshold":
            assert self.min_power is not None, ("Minimum power threshold "
                                                "is not defined.")
            # If minimum power threshold is not exceeded,
            # nobody gets to deploy geoengineering.
            if winner_power < self.min_power:
                G = 0.

            # If in the minimum power threshold scenario the geoengineering
            # deployment is positive, check that there is a unique coalition
            # with the highest share of global power. This is not required in
            # general, but simplifies things in the three-country model
            # considered in this paper.
            else:
                msg = "Several winning coalitions not allowed"
                assert self.coalition_powers.count(winner_power) == 1, msg

                msg = "Incorrect winner assignment"
                assert all(i <= winner_power for i in self.coalition_powers),\
                    msg

        return G

    @property
    def payoffs(self) -> dict:
        G = self.geo_deployment_level
        payoffs = {}

        names = [country.name for country in self.all_countries]
        payoffs = [country.payoff(G) for country in self.all_countries]

        return dict(zip(names, payoffs))
