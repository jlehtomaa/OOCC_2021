"""
Authors: Jere Lehtomaa & Daniel Heyen.

Replicates all results in Heyen & Lehtomaa (2021): Solar Geoengineering
Governance: A Dynamic Framework of Farsighted Coalition Formation.
"""


import pandas as pd
from lib.country import Country
from lib.coalition import Coalition
from lib.state import State
from lib.probabilities import TransitionProbabilities
from lib.mdp import MDP
from lib.utils import (derive_effectivity,
                       get_payoff_matrix,
                       get_geoengineering_levels,
                       verify_equilibrium,
                       write_latex_tables)


def run_experiment(config):

    # 1. Initialize all countries according to the current experiment.
    all_countries = []
    for player in config["players"]:
        country = Country(
            name=player,
            base_temp=config["base_temp"][player],
            delta_temp=config["delta_temp"][player],
            ideal_temp=config["ideal_temp"][player],
            m_damage=config["m_damage"][player],
            power=config["power"][player]
        )
        all_countries.append(country)

    # 2. Initialize the coalitions and states.
    # coalition_map = {state name: coalition structure}
    W, T, C = all_countries
    coalition_map = {
        '( )': [Coalition([W]), Coalition([T]), Coalition([C])],
        '(TC)': [Coalition([W]), Coalition([T, C])],
        '(WC)': [Coalition([T]), Coalition([W, C])],
        '(WT)': [Coalition([C]), Coalition([W, T])],
        '(WTC)': [Coalition([W, T, C])]
    }

    states = [State(
        name=name,
        coalitions=coalition_map[name],
        all_countries=all_countries,
        power_rule=config["power_rule"],
        min_power=config["min_power"]
    ) for name in config["state_names"]]

    payoffs = get_payoff_matrix(states=states, columns=config["players"])
    geoengineering = get_geoengineering_levels(states=states)

    # 3. Read in the strategy profile excel table.
    # Note that the effectivity correspondence is indirectly
    # deduced from the raw excel sheet by assuming that (player, approval)
    # pairs that are not specified are not part of the approval committee.
    # Then, once the effectivity correspondence is derived from the raw
    # excel input, all missing values are filled with zeros.
    excel_file = config["strategy_table_path"] + config["strategy_table_name"]
    strategy_df = pd.read_excel(excel_file, header=[0, 1], index_col=[0, 1, 2])

    effectivity = derive_effectivity(df=strategy_df,
                                     players=config["players"],
                                     states=config["state_names"])
    strategy_df.fillna(0., inplace=True)

    # 4. Derive the transition probability matrices
    # based on the strategy profiles of players.
    transition_probabilities = TransitionProbabilities(
                                df=strategy_df,
                                effectivity=effectivity,
                                players=config["players"],
                                states=config["state_names"],
                                protocol=config["protocol"],
                                unanimity_required=config["unanimity_required"]
                                )
    P, P_proposals, P_approvals = transition_probabilities.get_probabilities()

    # 5. Define the Markov Decision Process, and solve the corresponding
    # linear system of equations for all players, given their static
    # payoffs, transition probabilities across states, and their rate
    # of discounting.
    mdp = MDP(n_states=len(states),
              transition_probs=P,
              discounting=config["discounting"])

    V = pd.DataFrame(index=config["state_names"], columns=config["players"])
    for player in config["players"]:
        V.loc[:, player] = mdp.solve_value_func(payoffs.loc[:, player])

    # 6. Return all values calculated for this experiment and
    # pass them to the final check of strategy consistency.

    experiment_results = dict(
        experiment_name=config["experiment_name"],
        V=V, P=P, geoengineering=geoengineering, payoffs=payoffs,
        P_proposals=P_proposals, P_approvals=P_approvals,
        players=config["players"], state_names=config["state_names"],
        effectivity=effectivity, strategy_df=strategy_df
    )

    return experiment_results


def main():
    players = ["W", "T", "C"]
    n_players = len(players)

    base_config = dict(
        base_temp={"W": 21.5, "T": 14.0, "C": 11.5},
        ideal_temp={player: 13. for player in players},
        delta_temp={player: 3. for player in players},
        power={player: 1/n_players for player in players},
        protocol={player: 1/n_players for player in players},
        discounting=0.99,
        players=players,
        state_names=['( )', '(TC)', '(WC)', '(WT)', '(WTC)'],
        strategy_table_path="./strategy_tables/"
    )

    experiment_configs = {
        "weak_governance": dict(
            experiment_name="main_text_weak_governance",
            m_damage={player: 1. for player in players},
            power_rule="weak_governance",
            min_power=None,
            strategy_table_name="weak_governance.xlsx",
            unanimity_required=True
        ),

        "power_threshold": dict(
            experiment_name="main_text_power_threshold",
            m_damage={player: 1. for player in players},
            power_rule="power_threshold",
            min_power=0.5,
            strategy_table_name="power_threshold.xlsx",
            unanimity_required=True
        ),

        "power_threshold_extra_1": dict(
            experiment_name="supplementary_with_unanimity",
            m_damage={"W": 0.75, "T": 1.25, "C": 1.},
            power_rule="power_threshold",
            min_power=0.5,
            strategy_table_name="power_threshold.xlsx",
            unanimity_required=True
        ),

        "power_threshold_extra_2": dict(
            experiment_name="supplementary_without_unanimity",
            m_damage={"W": 0.75, "T": 1.25, "C": 1.},
            power_rule="power_threshold",
            min_power=0.5,
            strategy_table_name="power_threshold_no_unanimity.xlsx",
            unanimity_required=False
        )
    }

    for experiment, experiment_config in experiment_configs.items():

        config = {**base_config, **experiment_config}
        experiment_results = run_experiment(config)
        success, message = verify_equilibrium(experiment_results)

        try:
            assert success
        except AssertionError:
            print(message)

        write_latex_tables(experiment_results,
                           variables=["V", "payoffs", "P", "geoengineering"])

        print("Experiment:", experiment)
        print("Status:", message)
        print(10 * "-")


if __name__ == "__main__":
    main()
