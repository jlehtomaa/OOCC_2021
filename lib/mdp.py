import numpy as np
import pandas as pd


class MDP:
    """
    General Markov Decision Process placeholder.

    Arguments:
      n_states: Number of possible states to be considered.
      transition_probs: n_states * n_states matrix of probabilities.
      discounting: Discounting parameter.
    """
    def __init__(self,
                 n_states: int,
                 transition_probs: np.ndarray,
                 discounting: float):

        self.n_states = n_states
        self.transition_probs = transition_probs
        self.discounting = discounting

    def solve_value_func(self,
                         payoffs) -> np.ndarray:
        """
        Solve the linear system for an individual player.

        Arguments:
          player: Name of the player.
          payoffs: Matrix of payoffs, with states as rows and all players
                   as columns.
        """

        A = np.zeros((self.n_states, self.n_states))
        b = np.zeros(self.n_states)

        # For every state...
        for s in range(self.n_states):
            # ...consider all possible next states
            P = self.transition_probs
            for s_next, prob in enumerate(P[s, :]):
                A[s][s_next] = self.discounting * prob

        A -= np.eye(self.n_states)
        b = -(1-self.discounting) * payoffs
        x = np.linalg.solve(A, b)

        assert np.allclose(np.dot(A, x), b)

        return x
