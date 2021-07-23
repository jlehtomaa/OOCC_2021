class Country:
    """ Represents an individual player in the model.

    Arguments:
        name: Player name ('W', 'T', or 'C' used in the paper).
        base_temp: Preindustrial baseline temperature. T^base in eq. (B.1
        delta_temp: Climate-induced temperature change. Delta in eq. (B.1).
        ideal_temp: Ideal temperature. T^ideal in eq. (B.3).
        m_damage: Marginal damage term. Param. d in eq. (B.3).
        power: Country's share of global power. Param. gamma in eq. (B.6).
    """

    def __init__(self,
                 name: str,
                 base_temp: float,
                 delta_temp: float,
                 ideal_temp: float,
                 m_damage: float,
                 power: float,
                 ):

        assert m_damage >= 0., "Marginal damage cannot be negative"
        assert 0. <= power <= 1., "Power must be in [0,1]"

        self.name = name
        self.base_temp = base_temp
        self.delta_temp = delta_temp
        self.ideal_temp = ideal_temp
        self.m_damage = m_damage
        self.power = power

    @property
    def climate_change_temp(self) -> float:
        """Eq. (B.1).
        Current temperature with zero geoengineering deployment.
        """
        return self.base_temp + self.delta_temp

    @property
    def ideal_geoengineering_level(self) -> float:
        """Parameter alpha in eq. (B.4)."""
        return self.climate_change_temp - self.ideal_temp

    @property
    def climate_change_damage(self) -> float:
        """ Damages with zero geoengineering deployment.
        Corresponds to parameter K in eq. (B.3).
        """
        return self.m_damage * (self.climate_change_temp-self.ideal_temp) ** 2

    @property
    def weighted_damage(self) -> float:
        """Power-weighter marginal damage.
        Corresponds to parameter eta in eq. (B.7).
        """
        return self.power * self.m_damage

    def damage(self, G: float) -> float:
        """ Climate damages with geoengineering.

        Damages are normalized such that damages with zero SG deployment
        are zero for all countries.

        Arguments:
            G: Current global geoengineering deployment.
        """
        deviation = self.ideal_geoengineering_level - G
        cc_damage = self.climate_change_damage

        return self.m_damage * deviation ** 2 - cc_damage

    def payoff(self, G: float) -> float:
        """ Eq. (B.3). Country's payoff function under geoengineering.

        Arguments:
            G: Current global geoengineering deployment.
        """
        return -self.damage(G)
