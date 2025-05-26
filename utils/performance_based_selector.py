import math

import numpy as np
import pymongo
from solutions.genetic import models_to_individuals, save_individuals
from runner.fair_evaluator import get_individuals_from_collection, get_scenarios_from_collection
from utils.scenario_utils import ResultOfStep
import copy
import itertools


GEN_DB_TO_USE = "extended_genetic_ind_actual"
SCENARIO_DB_TO_USE = "scenarios_ex_2"

client = pymongo.MongoClient("mongodb://localhost:27017/")


def select_top_x_individuals(gen_collection: str, scenario_collection: str, env_type: str, x: int, target_collection: str = None):
    individuals = get_individuals_from_collection(gen_collection, env_type, GEN_DB_TO_USE)
    scenarios = get_scenarios_from_collection(scenario_collection, env_type, SCENARIO_DB_TO_USE)

    if target_collection is None:
        target_collection = gen_collection + f"-top-{x}"

    accumulated_scores = {}
    for individual in individuals:
        accumulated_scores[individual.id] = 0

    for scenario in [s for s in scenarios]:
        for r in range(3):
            for individual in [_ for _ in individuals]:
                scenario_copy = copy.deepcopy(scenario)
                individual.set_specific_scenario(scenario_copy)
                while individual.scenario.is_alive:
                    individual.act()
                    if individual.steps_made >= 500:
                        individual.scenario.is_alive = False
                accumulated_scores[individual.id] =  accumulated_scores[individual.id] + individual.steps_made

    selected_individuals = []


    for i in sorted(accumulated_scores, key=accumulated_scores.get, reverse=True)[:x]:
        for ind in individuals:
            if ind.id == i:
                selected_individuals.append(ind)

    save_individuals(target_collection, selected_individuals, env_type)


def select_top_x_individuals_ex(gen_collection: str, scenario_collection: str, env_type: str, x: int, target_collection: str = None):
    individuals = get_individuals_from_collection(gen_collection, env_type, GEN_DB_TO_USE)
    scenarios = get_scenarios_from_collection(scenario_collection, env_type, SCENARIO_DB_TO_USE)

    if target_collection is None:
        target_collection = gen_collection + f"-top-{x}-won"

    accumulated_scores = {}
    for individual in individuals:
        # score cumulative_steps, cumulative_oot, cumulative_shots, won
        accumulated_scores[individual.id] = (0, 0, 0, 0)

    for scenario in [s for s in scenarios]:
        for r in range(3):
            for individual in [_ for _ in individuals]:
                scenario_copy = copy.deepcopy(scenario)
                individual.set_specific_scenario(scenario_copy)
                shots = 0
                oot = 0
                won = 0
                while individual.scenario.is_alive:
                    res = individual.act()
                    if res == ResultOfStep.SHOT_LION:
                        shots += 1
                    if individual.steps_made >= 1000:
                        oot = 1
                        individual.scenario.is_alive = False
                    if individual.scenario.is_won():
                        won += 1
                        individual.scenario.is_alive = False
                new_score = accumulated_scores[individual.id]
                accumulated_scores[individual.id] = (new_score[0] + individual.steps_made, new_score[1] + oot, new_score[2] + shots, new_score[3] + won)

    final_scores = {}
    for ind, score in accumulated_scores.items():
        avg_steps = score[0] / len(scenarios)
        avg_shots = score[2] / len(scenarios)
        final_score = math.sqrt(avg_steps) + score[1]/2 + avg_shots
        final_score += 2 * score[3]
        final_score = round(final_score, 2)
        final_scores[ind] = final_score

    selected_individuals = []

    for i in sorted(final_scores, key=final_scores.get, reverse=True)[:x]:
        for ind in individuals:
            if ind.id == i:
                selected_individuals.append(ind)

    save_individuals(target_collection, selected_individuals, env_type)


def main():
    # select_top_x_individuals("naive_simple_T30L30_10k_3-ft", "simple_T30L30", "simple", 25)
    # select_top_x_individuals("prog_simple_T30L30_5h_long+5k", "simple_T30L30", "simple", 25)
    # select_top_x_individuals("naive_simple_T20L40_10k_4-ft", "simple_T30L30", "simple", 25)
    # select_top_x_individuals("prog_simple_T20L40_5h_long+1k+5k", "simple_T30L30", "simple", 25)
    # select_top_x_individuals("prog_simple_T20L40_5h_long+10k", "simple_T30L30", "simple", 25)

    # select_top_x_individuals("normal_simple_T20L40_5h_long-ft", "simple_T20L40", "simple", 3) nah
    # select_top_x_individuals("naive_simple_T20L40_10k_4-ft", "simple_T20L40", "simple", 3)
    # select_top_x_individuals("prog_simple_T20L40_5h_long+10k", "simple_T20L40", "simple", 3)
    # select_top_x_individuals("prog_simple_T20L40_5h_long+1k+5k", "simple_T20L40", "simple", 3)

    # select_top_x_individuals("naive_simple_T30L30_10k_3-ft", "simple_T30L30", "simple", 3)
    # select_top_x_individuals("prog_simple_T30L30_5h_long+5k", "simple_T30L30", "simple", 3)
    # select_top_x_individuals("naive_simple_T20L40_10k_4-ft", "simple_T30L30", "simple", 3)
    # select_top_x_individuals("prog_simple_T20L40_5h_long+1k+5k", "simple_T30L30", "simple", 3)
    # select_top_x_individuals("prog_simple_T20L40_5h_long+10k", "simple_T30L30", "simple", 3)

    # for i in range(1,6):
        # select_top_x_individuals(f"naive_T20L30_{i}-ft", "simple_T30L30", "simple", 10)
        # select_top_x_individuals(f"normal_T20L30_{i}-ft", "simple_T30L30", "simple", 10)
    #     select_top_x_individuals(f"prog_T20L30_{i}-ft", "simple_T30L30", "simple", 10)
    #     select_top_x_individuals(f"prog_T20L30_{i}-ft+5k", "simple_T30L30", "simple", 10)

    # for i in range(2,10):
    #     try:
    # #         # select_top_x_individuals_ex(f"naive_ex_T30L30_10k_{i}-ft", "extended_T30L20", "extended", 10)
    # #         # select_top_x_individuals_ex(f"normal_ex_T30L20_{i}-ft", "extended_T30L20", "extended", 10)
    # #         # select_top_x_individuals_ex(f"prog_ex_T30L20_{i}_real-ft", "extended_T30L20", "extended", 10)
    #         select_top_x_individuals_ex(f"prog_ex_T30L20_{i}_real-ft+5k", "extended_T30L20", "extended", 10) #other db
    #     except ValueError:
    #         pass

    # select_top_x_individuals_ex(f"prog_ex_T30L20_1_real-ft+5k", "extended_T30L20", "extended", 10)  # other db
    # select_top_x_individuals_ex(f"prog_ex_T30L20_10_real-ft+5k-top-10-won", "extended_T30L20", "extended", 3)  # other db


    pass




if __name__ == '__main__':
    main()