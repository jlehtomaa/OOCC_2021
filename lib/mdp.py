import numpy as np
import pandas as pd


class MDP:
    """
    General Markov Decision Process placeholder.

    Arguments:
        n_states: Number of possible states in the system.
        transition_probs: n_states * n_states matrix of probabilities.
        discounting: Discounting (and farsightedness) parameter.

    Implementation of this class follows closely the structure in:
    http://aima.cs.berkeley.edu/python/mdp.html and
    https://github.com/akaAlbo/deeprlbootcamp/tree/master/lab1
    """
    def __init__(self,
                 n_states: int,
                 transition_probs: pd.DataFrame,
                 discounting: float):

        self.n_states = n_states
        self.transition_probs = transition_probs
        self.discounting = discounting

    def solve_value_func(self, payoffs: np.ndarray) -> np.ndarray:
        """ Solve the linear system of value functions
        for an individual player.

        Arguments:
            payoffs: A vector of payoffs size n_states for a single country.
        """

        A = np.zeros((self.n_states, self.n_states))
        b = np.zeros(self.n_states)
        P = self.transition_probs

        for state in range(self.n_states):
            for next_state, prob in enumerate(P.iloc[state, :]):
                A[state][next_state] = self.discounting * prob

        A -= np.eye(self.n_states)
        b = -(1-self.discounting) * payoffs
        x = np.linalg.solve(A, b)

        assert np.allclose(np.dot(A, x), b)

        return x
