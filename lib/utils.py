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


def derive_effectivity(df: pd.DataFrame, players: list,
                       states: list) -> np.ndarray:
    """ Defines the effectivity correspondence.

    For each possible proposer, every possible state transition, and
    every possible other player as a responder, the effectivity matrix
    has a value of 1 if that responder is in the approval committee, and
    0 otherwise.

    Note that the set of responders is the full set of players. That is, 
    we also consider cases where a proposer can "propose" a transition to
    itself. This is important, as in most settings countries might be able
    to unilaterally exit their current coalition structure.

    Arguments:
      df: A DataFrame instance containing the strategies of all players.
      players: A list (str) of all the players in the game.
      states: A list (str) of all the considered states of the system.

    Returns:
      effectivity: a boolean array of the shape
                   (n_players, n_players, n_states, n_states), corresponding
                   to (proposer, responder, current_state, next_state)
                   dimensions. Each entry tells whether the responder is
                   a member of the approval committee, when the proposer
                   suggests a transition from current_state to next_state.
    """
    n_players = len(players)
    n_states = len(states)

    effectivity = np.zeros((n_players, n_players, n_states, n_states))
    # The dimensions are: [proposer, responder, current_state, next_state]

    for prop_idx, proposer in enumerate(players):

        # Consider all possible current states and proposed
        # next states.
        for current_state_idx, current_state in enumerate(states):
            for next_state_idx, next_state in enumerate(states):
                for responder in players:
                    resp_idx = players.index(responder)

                    resp_val = df.loc[(current_state, 'Acceptance', responder),
                                      (f'Proposer {proposer}', next_state)]

                    # If the corresponding cell is not empty, the player
                    # is a member of the approval committee.
                    is_member = int(~np.isnan(resp_val))

                    effectivity[prop_idx, resp_idx, current_state_idx,
                                next_state_idx] = is_member

                # Trivially, the proposer must approve the transition,
                # and is therefore included in the effectivity correspondence.
                if current_state == next_state:
                    effectivity[prop_idx, prop_idx, current_state_idx,
                                next_state_idx] = 1

    return effectivity
