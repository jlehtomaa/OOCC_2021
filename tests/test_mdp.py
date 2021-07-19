import numpy as np
import pandas as pd
from lib.mdp import MDP


def test_solve_value_func():

    states = ['s1', 's2', 's3']
    players = ['A', 'B', 'C']
    transition_probs = np.eye(len(states))
    payoffs = pd.DataFrame(1234., index=states, columns=players)

    mdp = MDP(n_states=len(states),
              transition_probs=transition_probs,
              discounting=0.9)

    V = mdp.solve_value_func(player='A', payoffs=payoffs)

    assert np.isclose(V, 1234.).all()
