import numpy as np
import pandas as pd


def calculate_transition_probabilities(df: pd.DataFrame,
                                       effectivity: np.ndarray,
                                       players: list, states: list,
                                       protocol: dict) -> tuple:

    n_players = len(players)
    n_states = len(states)

    # Transition probability matrix of the Markov Decision Process.
    # Rows denote current states, and columns the next states. Values
    # are probabilities of the corresponding state transitions.
    P = np.zeros((n_states, n_states))

    # What is the probability that player i proposes moving from state
    # x to state y.
    P_proposals = np.zeros((n_players, n_states, n_states))

    # What is the probability that player j accepts the proposed
    # transition from state x to state y.
    P_approvals = np.zeros((n_players, n_states, n_states))

    for prop_idx, proposer in enumerate(players):
        for current_state_idx, current_state in enumerate(states):
            for next_state_idx, next_state in enumerate(states):

                # The subset of players with the power to
                # approve this transition.
                approval_committee = effectivity[prop_idx, :,
                                                 current_state_idx,
                                                 next_state_idx] == 1
                approvers = np.array(players)[approval_committee]

                # Probability that proposer gets chosen by the protocol and
                # proposes next_state while in current_state.
                P_proposal = df.loc[(current_state, 'Proposition', np.nan),
                                    (f'Proposer {proposer}', next_state)]

                P_proposed = protocol[proposer] * P_proposal
                P_proposals[prop_idx, current_state_idx,
                            next_state_idx] = P_proposal

                if current_state == next_state:
                    P_approved = 1.
                elif len(approvers) == 0:
                    P_approved = 0.
                else:
                    P_approved = np.prod(
                            df.loc[(current_state, 'Acceptance', approvers),
                                   (f'Proposer {proposer}', next_state)]
                                   )

                P_approvals[prop_idx, current_state_idx,
                            next_state_idx] = P_approved
                P_rejected = 1 - P_approved

                # If approved, state changes.
                P[current_state_idx][next_state_idx] += P_proposed * P_approved
                # Otherwise, state remains unchanged.
                P[current_state_idx][current_state_idx] += P_proposed *\
                    P_rejected

    assert np.isclose(np.sum(P, axis=1), 1).all()
    assert (0. <= P).all() and (P <= 1.).all()

    return (P, P_proposals, P_approvals)
