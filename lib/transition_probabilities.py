import numpy as np
import pandas as pd


class TransitionProbabilities:
    def __init__(self,
                 df: pd.DataFrame,
                 effectivity: np.ndarray,
                 players: list, states: list,
                 protocol: dict,
                 unanimity_required: bool):

        self.df = df
        self.effectivity = effectivity
        self.players = players
        self.states = states
        self.protocol = protocol
        self.unanimity_required = unanimity_required

        n_players = len(players)
        n_states = len(states)

        # Capital P's stand for probability matrices, lowercase p's for
        # scalar probability values.
        self.P = np.zeros((n_states, n_states))
        self.P_proposals = np.zeros((n_players, n_states, n_states))
        self.P_approvals = np.zeros((n_players, n_states, n_states))

    def get_probabilities(self):
        if self.unanimity_required:
            return self.transition_probabilities_with_unanimity()
        else:
            return self.transition_probabilities_without_unanimity()

    def safety_checks(self):
        assert np.isclose(np.sum(self.P, axis=1), 1).all()
        assert (0 <= self.P).all() and (self.P <= 1).all()
        assert (0 <= self.P_proposals).all() and (self.P_proposals <= 1).all()
        assert (0 <= self.P_approvals).all() and (self.P_approvals <= 1).all()

    def get_approval_committee(self, prop_idx: int, current_state_idx: int,
                               next_state_idx: int) -> np.ndarray:

        approval_committee = self.effectivity[prop_idx, :, current_state_idx,
                                              next_state_idx] == 1
        approvers = np.array(self.players)[approval_committee]
        return approvers

    def read_proposal_prob(self, proposer, current_state, next_state):
        probability = self.df.loc[(current_state, 'Proposition', np.nan),
                                  (f'Proposer {proposer}', next_state)]
        return probability

    def transition_probabilities_with_unanimity(self):

        for prop_idx, proposer in enumerate(self.players):
            for current_state_idx, current_state in enumerate(self.states):
                for next_state_idx, next_state in enumerate(self.states):

                    approvers = self.get_approval_committee(prop_idx,
                                                            current_state_idx,
                                                            next_state_idx)

                    # Probability that proposer proposes next_state while
                    # in current_state.
                    p_proposal = self.read_proposal_prob(proposer,
                                                         current_state,
                                                         next_state)

                    self.P_proposals[prop_idx, current_state_idx,
                                     next_state_idx] = p_proposal

                    # Maintaining status quo is trivially approved.
                    if current_state == next_state:
                        p_approved = 1.
                    # If the approval committee is empty, the state transition
                    # is impossible.
                    elif len(approvers) == 0:
                        p_approved = 0.
                    # Otherwise, the acceptance requires the unanimous approval
                    # of the approval committee.
                    else:
                        p_approved = np.prod(
                          self.df.loc[(current_state, 'Acceptance', approvers),
                                      (f'Proposer {proposer}', next_state)])

                    self.P_approvals[prop_idx, current_state_idx,
                                     next_state_idx] = p_approved
                    p_rejected = 1 - p_approved

                    # Probability that proposer is chosen by the protocol, AND
                    # proposes the transition current_state -> next_state.
                    p_proposed = self.protocol[proposer] * p_proposal

                    # If proposed and approved, state changes.
                    self.P[current_state_idx][next_state_idx] +=\
                        p_proposed * p_approved
                    # Otherwise, state remains unchanged.
                    self.P[current_state_idx][current_state_idx] +=\
                        p_proposed * p_rejected

        self.safety_checks()
        return (self.P, self.P_proposals, self.P_approvals)

    def list_members(self, state: str) -> list:
        """ Lists all the member countries of the existing coalition.

        list_current_members('(WTC)') returns ['W', 'T', 'C'].
        """
        return list(state[state.find("(")+1:state.find(")")])

    def transition_probabilities_without_unanimity(self):

        for prop_idx, proposer in enumerate(self.players):
            for current_state_idx, current_state in enumerate(self.states):

                # Countries that are members of the coalition in the current
                # state. NOTE: This rests on the assumption of out 3-player
                # game that only one non-singleton coalitoin can exist at
                # any time. For more players, this needs to be re-implemented
                # with some different logic.
                old_members = self.list_members(current_state)

                for next_state_idx, next_state in enumerate(self.states):

                    approvers = self.get_approval_committee(prop_idx,
                                                            current_state_idx,
                                                            next_state_idx)

                    new_members = [country for country
                                   in self.list_members(next_state)
                                   if country not in old_members]

                    # Probability that proposer proposes next_state while
                    # in current_state.
                    p_proposal = self.read_proposal_prob(proposer,
                                                         current_state,
                                                         next_state)

                    self.P_proposals[prop_idx, current_state_idx,
                                     next_state_idx] = p_proposal

                    # Maintaining status quo is trivially approved:
                    if current_state == next_state:
                        p_approved = 1.
                    # If the approval committee is empty, the state transition
                    # is impossible.
                    elif len(approvers) == 0:
                        p_approved = 0.
                    # If the approval committee only has one member, it can
                    # decide alone whether or not to approve the transition.
                    elif len(approvers) == 1:
                        p_approved = self.df.loc[
                                   (current_state, 'Acceptance', approvers),
                                   (f'Proposer {proposer}', next_state)].values
                    else:
                        # Check that all new members approve.
                        if len(new_members) == 0:
                            p_new_members_approve = 0.
                        elif len(new_members) == 1 and proposer in new_members:
                            p_new_members_approve = 1.
                        else:
                            p_new_members_approve = np.prod(self.df.loc[
                                    (current_state, 'Acceptance', new_members),
                                    (f'Proposer {proposer}', next_state)])

                        # Check that majority of old members approves.
                        if len(old_members) == 0:
                            p_old_members_approve = 0.
                        else:

                            probs = self.df.loc[
                                      (current_state, 'Acceptance', approvers),
                                      (f'Proposer {proposer}', next_state)]
                            p_old_members_approve =\
                                np.sum(probs) - np.prod(probs)

                        p_approved =\
                            p_new_members_approve * p_old_members_approve

                    self.P_approvals[prop_idx, current_state_idx,
                                     next_state_idx] = p_approved
                    p_rejected = 1 - p_approved

                    p_proposed = self.protocol[proposer] * p_proposal
                    # If approved, state changes.
                    self.P[current_state_idx][next_state_idx] +=\
                        p_proposed * p_approved
                    # Otherwise, state remains unchanged.
                    self.P[current_state_idx][current_state_idx] +=\
                        p_proposed * p_rejected

        self.safety_checks()
        return (self.P, self.P_proposals, self.P_approvals)


# def calculate_transition_probabilities(df: pd.DataFrame,
#                                        effectivity: np.ndarray,
#                                        players: list, states: list,
#                                        protocol: dict) -> tuple:

#     """ Translates the equilibrium strategies into transition
#     probabilities between different states.

#     Arguments:
#       df: A dataframe containing the strategy profiles of all players.
#       effectivity: The effectivity correspondence from derive_effectivity().
#       players: List (str) of all players in the game.
#       states: List (str) of all possible states of the system.
#       protocol: Dict with players as keys and probabilities of being chosen
#                 as the proposer as values.

#     Returns:
#       P: Size (n_states, n_states) matrix of transition probabilities of the
#          Markov Decision Process. Rows denote current states,
#          and columns the possible next states.
#       P_proposals: Size (n_players, n_states, n_states) array. Each entry is
#                    the probability that player i, IF chosen as proposer,
#                    suggests a move from state x to a new state y.
#       P_approvals: Size (n_players, n_states, n_states) array. Each entry is
#                    probability that the proposition by player i to move from
#                    state x to a new state y gets accepted by the
#                    approval committee.
#     """

#     n_players = len(players)
#     n_states = len(states)

#     # Capital P's stand for probability matrices, lowercase p's for
#     # scalar probability values.
#     P = np.zeros((n_states, n_states))
#     P_proposals = np.zeros((n_players, n_states, n_states))
#     P_approvals = np.zeros((n_players, n_states, n_states))

#     for prop_idx, proposer in enumerate(players):
#         for current_state_idx, current_state in enumerate(states):
#             for next_state_idx, next_state in enumerate(states):

#                 # The subset of players with the power to
#                 # approve this transition.
#                 approval_committee = effectivity[prop_idx, :,
#                                                  current_state_idx,
#                                                  next_state_idx] == 1
#                 approvers = np.array(players)[approval_committee]

#                 # Probability that proposer proposes next_state while
#                 # in current_state.
#                 p_proposal = df.loc[(current_state, 'Proposition', np.nan),
#                                     (f'Proposer {proposer}', next_state)]

#                 P_proposals[prop_idx, current_state_idx,
#                             next_state_idx] = p_proposal

#                 # Maintaining status quo is trivially approved.
#                 if current_state == next_state:
#                     p_approved = 1.
#                 # If the approval committee is empty, the state transition
#                 # is impossible.
#                 elif len(approvers) == 0:
#                     p_approved = 0.
#                 # Otherwise, the acceptance requires the unanimous approval
#                 # of the approval committee.
#                 else:
#                     p_approved = np.prod(
#                             df.loc[(current_state, 'Acceptance', approvers),
#                                    (f'Proposer {proposer}', next_state)])

#                 P_approvals[prop_idx, current_state_idx,
#                             next_state_idx] = p_approved
#                 p_rejected = 1 - p_approved

#                 # Probability that proposer is chosen by the protocol, AND
#                 # proposes the transition current_state -> next_state.
#                 p_proposed = protocol[proposer] * p_proposal

#                 # If proposed and approved, state changes.
#                 P[current_state_idx][next_state_idx] += p_proposed*p_approved
#                 # Otherwise, state remains unchanged.
#                 P[current_state_idx][current_state_idx] += p_proposed *\
#                     p_rejected

#     # Final safety checks.
#     assert np.isclose(np.sum(P, axis=1), 1).all()
#     assert (0. <= P).all() and (P <= 1.).all()
#     assert (0. <= P_proposals).all() and (P_proposals <= 1.).all()
#     assert (0. <= P_approvals).all() and (P_approvals <= 1.).all()

#     return (P, P_proposals, P_approvals)
