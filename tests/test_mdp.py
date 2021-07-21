import numpy as np
import pandas as pd
from lib.mdp import MDP


def test_solve_value_func():

    states = ['s1', 's2', 's3']
    transition_probs = pd.DataFrame(np.eye(len(states)), index=states,
                                    columns=states)
    payoffs = 1234. * np.ones(len(states))

    mdp = MDP(n_states=len(states),
              transition_probs=transition_probs,
              discounting=0.9)

    V = mdp.solve_value_func(payoffs=payoffs)

    assert np.isclose(V, 1234.).all()
