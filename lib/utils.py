import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
from lib.state import State


def get_payoff_matrix(states: List[State], columns: List[str]) -> pd.DataFrame:
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
        assert list(state.payoffs.keys()) == columns,\
            "Payoff matrix cols and payoff dict keys do not match!"
        payoffs_df.loc[state.name, :] = state.payoffs

    return payoffs_df


def get_geoengineering_levels(states: List[State]) -> pd.DataFrame:
    """
    Returns the geoengineering deployment level for a given state.

    Arguments:
        states: A list of State instances (all states considered in the game).
    """
    assert all(isinstance(state, State) for state in states)

    G = {}
    for state in states:
        G[state.name] = state.geo_deployment_level

    return pd.DataFrame.from_dict(G, orient='index', columns=["G"])


def list_members(state: str) -> List[str]:
    """ Lists all the member countries of the existing coalition.

    For instance, list_current_members('(WTC)') returns ['W', 'T', 'C'].
    For ( ), returns an empty list.
    """
    no_brackets = list(state[state.find("(")+1:state.find(")")])
    return [char for char in no_brackets if char != " "]


def get_approval_committee(effectivity: Dict[tuple, int], players: List[str],
                           proposer: str, current_state: str,
                           next_state: str) -> List[str]:
    """Returns the list of all players who belong to the approval committee
    when proposer proposes the transition (current_state) -> (next_state).
    
    Arguments:
        effectivity: The effectivity correspondence, from derive_effectivity().
        players: The list (string) of all countries in the game.
        proposer: The current proposer country.
        current_state: Current coalition structure of the game.
        next_state: The next coalition structure suggested by proposer.
    """

    comm = [player for player in players
            if effectivity[(proposer, current_state, next_state, player)] == 1]

    return comm


def derive_effectivity(df: pd.DataFrame, players: List[str],
                       states: List[str]) -> Dict[tuple, int]:
    """ Defines the effectivity correspondence from the strategy profiles.

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
        effectivity: a dictionary with keys being the 4-tuples
                     (proposer, current_state, next_state, responder), and
                     the value being a boolean 0 or 1. Each entry tells
                     whether the responder is a member of the approval
                     committee, when the proposer suggests a transition from
                     the current_state to next_state.
    """

    effectivity = {}

    for proposer in players:
        for current_state in states:
            for next_state in states:
                for responder in players:

                    # If the corresponding 'acceptance' cell is not empty,
                    # the player is a member of the approval committee.
                    resp_val = df.loc[(current_state, 'Acceptance', responder),
                                      (f'Proposer {proposer}', next_state)]
                    is_member = int(~np.isnan(resp_val))

                    idx = (proposer, current_state, next_state, responder)
                    effectivity[idx] = is_member

                # Trivially, the proposer must approve the transition,
                # and is therefore included in the effectivity correspondence.
                # However, for convenience, we only include the proposer
                # explicitly in the strategy table when the proposer is the
                # only approval committee member, and thus can approve
                # the proposed transition without consulting others.

                # For every possible proposer, it is always possible to
                # maintain the status quo without the approval of others.
                # Therefore, for such a transition, check that the current
                # proposer is the only member in the effectivity
                # correspondence. Similarly, any country is allowed to
                # walk out of its existing coalition.
                if current_state == next_state or is_uniform_breakout(
                                                        proposer,
                                                        current_state,
                                                        next_state):

                    committee = get_approval_committee(effectivity, players,
                                                       proposer, current_state,
                                                       next_state)
                    assert [proposer] == committee

    return effectivity


def is_uniform_breakout(proposer: str, current_state: str,
                        next_state: str) -> bool:
    """Check if the current transition corresponds to the proposer alone
    walking out of an existing coalition. Such a move is always allowed, 
    and needs not be approved by any other players.
    
    Arguments:
        proposer: Name of the current proposer. E.g., 'T'.
        current_state: Current coalition structure. E.g., '(WTC)'.
        next_state: Proposed next coalition structure. E.g., '(WC)'.

    For instance: 'T' proposing '(WTC)' -> '(WC)' returns True.
    """

    current_members = list_members(current_state)
    next_members = list_members(next_state)

    # If breakout from grand coalition.
    if len(current_members) == 3 and len(next_members) == 2:
        if proposer in current_members and proposer not in next_members:
            return True

    # If breakout from a coalition of 2 players to all singletons.
    elif len(current_members) == 2 and len(next_members) == 0:
        if proposer in current_members:
            return True
    else:
        return False


def verify_proposals(players: List[str], states: List[str],
                     P_proposals: Dict[tuple, float],
                     P_approvals: Dict[tuple, float],
                     V: pd.DataFrame) -> Tuple[bool, str]:
    """Checks that the proposal strategies of all players constitute a
    valid equilibrium, as specified in Condition 1 in section A.5.
    
    Arguments:
        players: A list of all countries in the game.
        states: A list of all possible states in the system.
        P_proposals: A dictionary with keys determined by triplets
                     (i, x, y). Each value is the probability that player i,
                     IF chosen as proposer, suggests a move from the current
                     state x to a new state y.
        P_approvals: A dictionary with keys determined by triplets
                     (i, x, y). Each value is the probability that the
                     transition proposed by player i, to move from current
                     state x to a new state y, gets accepted by the
                     approval committee.
        V: A dataframe containing the long-run expected payoff for all
           players in all states.
    """

    for proposer in players:
        for current_state in states:

            # All next states for which the proposer attaches
            # a positive proposition probability while in current_state.
            pos_prob_next_states = []

            # Expectation of the proposition value:
            # E = p_accepted * V_next + p_rejected * V_current
            expected_values = {}

            for next_state in states:

                p_proposed = P_proposals[(proposer, current_state,
                                         next_state)]

                if p_proposed > 0.:
                    pos_prob_next_states.append(next_state)

                # Probability that the approval committee accepts.
                p_approved = P_approvals[(proposer, current_state,
                                         next_state)]
                p_rejected = 1 - p_approved

                V_current = V.loc[current_state, proposer]
                V_next = V.loc[next_state, proposer]
                expected_values[next_state] =\
                    p_approved * V_next + p_rejected * V_current

            # Next state(s) that give the highest possible expected
            # long-run payoff.
            argmaxes = [key for key, val in expected_values.items()
                        if np.isclose(val, max(expected_values.values()),
                        atol=1e-12)]

            try:
                # Any state with a positive proposal probability must be one
                # of the best alternatives.
                assert set(pos_prob_next_states).issubset(argmaxes)
            except AssertionError:
                error_msg = (
                         f"Proposal strategy error with player {proposer}! "
                         f"In state {current_state}, positive probability "
                         f"on state(s) {pos_prob_next_states}, but the argmax "
                         f"states are: {argmaxes}. \n"
                         f"The value functions V are: \n"
                         f"{V}"
                         )
                return False, error_msg

    return True, "Test passed."


def verify_approvals(players: List[str], states: List[str],
                     effectivity: Dict[tuple, int], V: pd.DataFrame,
                     strategy_df: pd.DataFrame) -> Tuple[bool, str]:
    """Checks that the approval strategies of all players constitute a
    valid equilibrium, as specified in Condition 2 in section A.5.
    
    Arguments:
        players: A list of all countries in the game.
        states: A list of all possible states in the system.
        effectivity: The effectivity correspondence, from derive_effectivity().
        V: A dataframe containing the long-run expected payoff for all
           players in all states.
        strategy_df: A dataframe containing the strategies of all players.
    """

    for proposer in players:
        for current_state in states:
            for next_state in states:

                # Approval committee for this transition.
                approvers = get_approval_committee(
                    effectivity, players, proposer, current_state, next_state)
                
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


def verify_equilibrium(result: Dict[str, Any]):
    """Checks that the experiment results and strategy profiles are a
    valid equilibrium.
    
    Arguments:
        results: A dictionary from main.run_experiment().
    """

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
        messages = [check[1] for check in [proposals_ok, approvals_ok]
                    if not check[0]]

        return False, '\n'.join(messages)


def write_result_tables_to_latex(result: Dict[str, Any], variables: List[str],
                                 results_path: str = "./results",
                                 float_format: str = "%.5f") -> None:
    """Writes experiment results as .tex tables.
    
    Arguments:
        results: A dictionary from main.run_experiment().
        variables: A list of items in results.keys() to store.
        results_path: Folder to store the .tex files in.
        float_format: How many digits to include in the .tex tables.
    """

    experiment = result['experiment_name']
    for variable in variables:

        path = f"{results_path}/{variable}_{experiment}.tex"
        result[variable].to_latex(buf=path, float_format=float_format,
                                  caption=f"{experiment}: {variable}")
