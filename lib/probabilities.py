import numpy as np
import pandas as pd
import warnings
from typing import List, Dict
from lib.utils import get_approval_committee, list_members
from lib.errors import ApprovalCommitteeError


class TransitionProbabilities:
    """ Translates the equilibrium strategies of countries into transition
    probabilities between different states.

    Arguments:
        df: A dataframe containing the strategy profiles of all players.
        effectivity: The effectivity correspondence calculated in
                     lib.utils.derive_effectivity().
        players: List (str) of all players in the game.
        states: List (str) of all possible states of the system.
        protocol: Dict with player names as keys and probabilities
                  of being chosen as the proposer as values.
        unanimity_required: A boolean value to indicate whether the
                approval committee needs to be perfectly unanimous for a
                proposition to pass. If False, a simple majority is enough.

    Returns:
        P: Size (n_states, n_states) matrix of transition probabilities of the
           Markov Decision Process. Rows denote current states,
           and columns the possible next states.
        P_proposals: A dictionary with keys determined by triplets
                     (i, x, y). Each value is the probability that player i,
                     IF chosen as proposer, suggests a move from the current
                     state x to a new state y.
        P_approvals: A dictionary with keys determined by triplets
                     (i, x, y). Each value is the probability that the
                     transition proposed by player i, to move from current
                     state x to a new state y, gets accepted by the
                     approval committee.
    """
    def __init__(self,
                 df: pd.DataFrame,
                 effectivity: Dict[tuple, int],
                 players: List[str],
                 states: List[str],
                 protocol: Dict[str, float],
                 unanimity_required: bool):

        self.df = df
        self.effectivity = effectivity
        self.players = players
        self.states = states
        self.protocol = protocol
        self.unanimity_required = unanimity_required

        # Notation: Capital P's stand for probability matrices,
        # lowercase p's for scalar probability values.
        self.P = pd.DataFrame(0., index=states, columns=states)
        self.P_proposals = {}
        self.P_approvals = {}

    def get_probabilities(self):
        if self.unanimity_required:
            return self.transition_probabilities_with_unanimity()
        else:
            return self.transition_probabilities_without_unanimity()

    def safety_checks(self):
        """Check that all computed values are valid probabilities."""
        # All rows in the state transition probability matrix sum up to one.
        assert np.isclose(self.P.sum(axis=1), 1.).all()

        # All probabilities are in [0, 1].
        assert (0. <= self.P.values).all() and (self.P.values <= 1.).all()
        assert all(0. <= val <= 1. for val in self.P_proposals.values())
        assert all(0. <= val <= 1. for val in self.P_approvals.values())

    def read_proposal_prob(self, proposer: str, current_state: str,
                           next_state: str) -> float:
        """Reads an individual proposal entry from the strategy table."""
        probability = self.df.loc[(current_state, 'Proposition', np.nan),
                                  (f'Proposer {proposer}', next_state)]
        return probability

    def read_approval_probs(self, approvers: List[str], proposer: str,
                            current_state: str, next_state: str) -> float:
        """Reads the acceptance probabilities for all members in approvers."""
        probability = self.df.loc[(current_state, 'Acceptance', approvers),
                                  (f'Proposer {proposer}', next_state)]
        return probability

    def empty_approval_committee_warning(self, indx: tuple):
        msg = f"Empty appoval committee for {indx[0]}: {indx[1]} -> {indx[2]}"
        warnings.warn(msg)

    def transition_probabilities_with_unanimity(self):
        """Calculate transition probabilities with a perfectly unanimous
        approval committee.
        """
        
        for proposer in self.players:
            for current_state in self.states:
                for next_state in self.states:

                    indx = (proposer, current_state, next_state)
                    approvers = get_approval_committee(
                        self.effectivity, self.players, *indx)

                    # Probability that the current proposer proposes
                    # next_state while in current_state.
                    p_proposal = self.read_proposal_prob(*indx)
                    self.P_proposals[indx] = p_proposal

                    # If the approval committee is empty, the state transition
                    # is impossible. This should not really happen in the
                    # scenarios considered here. One could add cases where
                    # some transitions are forbidden.
                    if len(approvers) == 0:
                        p_approved = 0.
                        self.empty_approval_committee_warning(indx)

                    # The probability of a transition being approved equals
                    # the probability that all approval committee members
                    # approve unanimously.
                    elif 1 <= len(approvers) <= 2:
                        probs = self.read_approval_probs(approvers, *indx)
                        p_approved = np.prod(probs)
                        # print(*indx, approvers, probs.values, p_approved)
                    else:
                        raise ApprovalCommitteeError(indx)

                    self.P_approvals[indx] = p_approved
                    p_rejected = 1 - p_approved

                    # Probability that proposer is chosen by the protocol, AND
                    # proposes the transition current_state -> next_state.
                    p_proposed = self.protocol[proposer] * p_proposal

                    # If proposed and approved, state changes.
                    self.P.loc[current_state, next_state] +=\
                        p_proposed * p_approved
                    # Otherwise, state remains unchanged.
                    self.P.loc[current_state, current_state] +=\
                        p_proposed * p_rejected

        self.safety_checks()
        return (self.P, self.P_proposals, self.P_approvals)

    def transition_probabilities_without_unanimity(self):
        """Calculate transition probabilities such that the approval is
        required from all new members, but only from the majority of
        existing members.
        """

        for proposer in self.players:
            for current_state in self.states:
                for next_state in self.states:
                    indx = (proposer, current_state, next_state)

                    approvers = get_approval_committee(
                        self.effectivity, self.players, *indx)

                    # Proposal probability:
                    # ---------------------

                    # Probability that proposer proposes next_state while
                    # in current_state.
                    p_proposal = self.read_proposal_prob(*indx)
                    self.P_proposals[indx] = p_proposal

                    # Approval probabilities:
                    # ----------------------

                    # If the approval committee is empty, the state transition
                    # is impossible. This should not really happen in the
                    # scenarios considered here. One could add cases where
                    # some transitions are forbidden.
                    if len(approvers) == 0:
                        p_approved = 0.
                        self.empty_approval_committee_warning(indx)

                    # If the approval committee only has one member, it can
                    # decide alone whether or not to approve the transition
                    # This covers both maintaining status quo, unilateral
                    # breakout, and cases such as W proposing ( ) -> (WC),
                    # where C is the only one who needs to approve.
                    elif len(approvers) == 1:
                        p_approved = self.read_approval_probs(
                            approvers, *indx).values
                        # print(*indx, approvers, p_approved)

                    # For a larger approval committee, we need to consider
                    # the cases where majority approval committee can
                    # validate state transitions.
                    else:
                        assert len(approvers) == 2
                        current_members = list_members(current_state)
                        next_members = list_members(next_state)

                        new_members = [country for country in next_members
                                       if country not in current_members]

                        current_non_proposer_members = [
                            country for country in current_members
                            if country != proposer]

                        new_non_proposer_members = [
                            country for country in new_members
                            if country != proposer]

                        if new_non_proposer_members:
                            # CASE 1:
                            # If there are new non-proposer members joining
                            # the new coalition, and the proposer is not
                            # an existing member, all of the new non-proposer
                            # members approve the transition.
                            # E.g., W proposing ( ) -> (TC) or ( ) -> (WTC)
                            # must be approved by both T and C.

                            # If there are new non-proposer members joining
                            # the new coalition, and the proposer is one of
                            # the existing members AND a member in the new
                            # coalition that forms, all new non-proposer
                            # members must approve the transition.
                            # E.g., W proposing (WC) -> (WT) or (WC) -> (WTC)
                            # must be approved by T but not C.
                            if (proposer not in current_members) or\
                                (proposer in current_members and
                                    proposer in next_members):
                                probs = self.read_approval_probs(
                                    new_non_proposer_members, *indx)
                                p_approved = np.prod(probs)
                                # print(*indx, new_non_proposer_members,
                                #       probs.values, p_approved)

                            # CASE 2:
                            # If there are new non-proposer members joining
                            # the new coalition, and the proposer is one of
                            # the existing members but not a member in the
                            # new coalition that forms, all countries in the
                            # new coalition must approve the transition.
                            # E.g., W proposing (WC) -> (TC) or (WT) -> (TC)
                            # must be approved by both T and C.
                            elif (proposer in current_members) and\
                                 (proposer not in next_members):
                                probs = self.read_approval_probs(
                                    next_members, *indx)
                                p_approved = np.prod(probs)
                                # print(*indx, next_members, probs.values,
                                #       p_approved)
                            else:
                                raise ApprovalCommitteeError(indx)

                        # CASE 3:
                        # If there are no new non-proposer members,
                        # at least one existing member must approve the
                        # proposed transition.
                        # E.g., W proposing (TC) -> ( ) or (TC) -> (WC)
                        # or (WTC) -> ( ) or (WTC) -> (WC) can be approved by
                        # either T or C, or W proposing (WTC)
                        elif not new_non_proposer_members:
                            probs = self.read_approval_probs(
                                    current_non_proposer_members, *indx)
                            p_approved = np.sum(probs) - np.prod(probs)
                            # print(*indx, current_non_proposer_members,
                            #       probs.values, p_approved)
                        else:
                            raise ApprovalCommitteeError(indx)

                    self.P_approvals[indx] = p_approved
                    p_rejected = 1 - p_approved

                    p_proposed = self.protocol[proposer] * p_proposal
                    # If approved, state changes.
                    self.P.loc[current_state, next_state] +=\
                        p_proposed * p_approved
                    # Otherwise, state remains unchanged.
                    self.P.loc[current_state, current_state] +=\
                        p_proposed * p_rejected

        self.safety_checks()
        return (self.P, self.P_proposals, self.P_approvals)
