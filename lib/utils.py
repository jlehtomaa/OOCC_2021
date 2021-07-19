import numpy as np
import pandas as pd
from lib.state import State


def get_payoff_matrix(states: list, columns: list) -> pd.DataFrame:
    """
    Calculate a payoff matrix for all states and countries in the game.

    Arguments:
      states: A list of State instances (all states considered in the game).
      columns: List of all player names (strings) included in the game.
    """
    assert all(isinstance(state, State) for state in states)

    state_names = [state.name for state in states]
    payoffs_df = pd.DataFrame(index=state_names, columns=columns,
                              dtype=np.float64)

    for state in states:
        payoffs_df.loc[state.name, :] = state.payoffs

    return payoffs_df


def get_geoengineering_levels(states: list) -> dict:
    """
    Returns the geoengineering deployment level for a given state.

    Arguments:
      states: A list of State instances (all states considered in the game).
    """
    assert all(isinstance(state, State) for state in states)

    G = {}
    for state in states:
        G[state.name] = state.geo_deployment_level

    return G
