import numpy as np
import pandas as pd
from lib.state import State


def get_payoff_matrix(states: list, columns: list) -> pd.DataFrame:
    """
    Calculate a payoff matrix for all states and countries in the game.

    Arguments:
        states: A list of State instances (all states considered in the game).
        columns: List (str) of all player names included in the game.
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

    effectivity = {}
    # The dimensions are: [proposer, responder, current_state, next_state]

    for proposer in players:
        for current_state in states:
            for next_state in states:
                for responder in players:

                    resp_val = df.loc[(current_state, 'Acceptance', responder),
                                      (f'Proposer {proposer}', next_state)]

                    # If the corresponding cell is not empty, the player
                    # is a member of the approval committee.
                    is_member = int(~np.isnan(resp_val))

                    idx = (proposer, current_state, next_state, responder)
                    effectivity[idx] = is_member

                # Trivially, the proposer must approve the transition,
                # and is therefore included in the effectivity correspondence.
                if current_state == next_state:
                    effectivity[(proposer, proposer, current_state, next_state)] = 1

    return effectivity


def verify_proposals(players, states, P_proposals, P_approvals, V):

    for proposer_idx, proposer in enumerate(players):
        for current_state_idx, current_state in enumerate(states):

            # All next states for which the proposer attaches
            # a positive proposition probability.
            P_prop_pos_states = []

            # Expectation of the proposition value:
            # E = P_accept * V_next + P_reject * V_current
            expected_values = {}

            for next_state_idx, next_state in enumerate(states):

                p_proposed = P_proposals[proposer_idx, current_state_idx,
                                         next_state_idx]

                if p_proposed > 0.:
                    P_prop_pos_states.append(next_state)

                p_approved = P_approvals[proposer_idx, current_state_idx,
                                         next_state_idx]
                p_rejected = 1 - p_approved

                V_current = V.loc[current_state, proposer]
                V_next = V.loc[next_state, proposer]
                expected_values[next_state] =\
                    p_approved * V_next + p_rejected * V_current

            argmaxes = [key for key, val in expected_values.items()
                        if np.isclose(val, max(expected_values.values()),
                        atol=1e-12)]

            try:
                assert set(P_prop_pos_states).issubset(argmaxes)
            except AssertionError:
                error_msg = (
                         f"Proposal strategy error with player {proposer}! "
                         f"In state {current_state}, positive probability "
                         f"on state(s) {P_prop_pos_states}, but the argmax "
                         f"states are: {argmaxes}. \n"
                         f"The value functions V are: \n"
                         f"{V}"
                         )
                return False, error_msg

    return True, "Test passed."


def verify_approvals(players, states, effectivity, V, strategy_df):

    # Consider all proposers one by one.
    for prop_idx, proposer in enumerate(players):
        for current_state_idx, current_state in enumerate(states):

            for next_state_idx, next_state in enumerate(states):
                # For all possible state transitions, get the
                # countries whose approval is needed.
                approval_committee_mask = effectivity[
                                            prop_idx, :,
                                            current_state_idx,
                                            next_state_idx] == 1
                approvers = np.array(players)[approval_committee_mask]

                for approver in approvers:

                    V_current = V.loc[current_state, approver]
                    V_next = V.loc[next_state, approver]
                    p_approve = strategy_df.loc[
                                    (current_state, 'Acceptance', approver),
                                    (f'Proposer {proposer}', next_state)]

                    if np.isclose(V_next, V_current, atol=1e-12):
                        passed = (0. <= p_approve <= 1.)
                    elif V_next > V_current:
                        passed = (p_approve == 1.)
                    elif V_next < V_current:
                        passed = (p_approve == 0.)
                    else:
                        msg = 'Unknown error during approval consistency check'
                        raise ValueError(msg)

                    if not passed:
                        error_msg = (
                            f"Approval strategy error with player {approver}! "
                            f"When player {proposer} proposes the transition "
                            f"{current_state} -> {next_state}, the values are "
                            f"V(current) = {V_current:.5f} "
                            f"and V(next) = {V_next:.5f}, "
                            f"but approval probability is {p_approve}."
                            )
                        return False, error_msg

    return True, "Test passed."


def verify_equilibrium(result):

    proposals_ok = verify_proposals(players=result["players"],
                                    states=result["state_names"],
                                    P_proposals=result["P_proposals"],
                                    P_approvals=result["P_approvals"],
                                    V=result["V"])

    approvals_ok = verify_approvals(players=result["players"],
                                    states=result["state_names"],
                                    effectivity=result["effectivity"],
                                    V=result["V"],
                                    strategy_df=result["strategy_df"])

    if proposals_ok[0] and approvals_ok[0]:
        return True, "All tests passed."
    else:
        msg = [check[1] for check in [proposals_ok, approvals_ok]
               if not check[0]]

        return False, '\n'.join(msg)


def write_result_tables_to_latex(result, variables, results_path="./results",
                                 float_format="%.5f"):

    experiment = result['experiment_name']
    for variable in variables:

        path = f"{results_path}/{variable}_{experiment}.tex"
        result[variable].to_latex(buf=path, float_format=float_format,
                                  caption=f"{experiment}: {variable}")
