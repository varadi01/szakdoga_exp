from solutions.genetic import Step, TileState, Action, Environment, ActionHolder, save_individuals
from runner.fair_evaluator import get_individuals_from_collection
from random import choice
import pymongo

GEN_DB_TO_USE = "extended_genetic_ind"

client = pymongo.MongoClient("mongodb://localhost:27017/")

def optimize(source_collection: str, number_of_optimizations):
    """only handles simple game"""
    individuals = get_individuals_from_collection(source_collection, "simple", GEN_DB_TO_USE)
    print(len(individuals))

    for individual in individuals:
        corrections = number_of_optimizations
        for action in individual.known_actions.actions:
            bad, fix = is_suboptimal(action)
            if bad:
                corrections -= 1
                action.step = fix
            if corrections == 0:
                break

    target_collection = f"{source_collection}-fix-{number_of_optimizations}"
    save_individuals(target_collection, individuals, "simple")



def is_suboptimal(action: Action) -> tuple[bool, Step]:
    trees = []
    env = action.env.get_as_list()
    for i in range(4):
        if env[i] == 1:
            trees.append(Step(i))
    if action.step in trees or len(trees) == 0:
        return False, None
    return True, choice(trees)

#todo use, and ex

# optimize("naive_T20L30_3-ft-top-10", 5)
# optimize("normal_T20L30_4-ft-top-10", 5)
# optimize("prog_T20L30_4-ft-top-10", 5)
# optimize("prog_T20L30_5-ft+5k-top-10", 5)
#
# optimize("naive_T30L30_4-ft-top-10", 5)
# optimize("normal_T30L30_1-ft-top-10", 5)
# optimize("prog_T30L30_1-ft-top-10", 5)
# optimize("prog_T30L30_1-ft+5k-top-10", 5)

optimize("naive_T20L30_3-ft-top-10", 10)
optimize("normal_T20L30_4-ft-top-10", 10)
optimize("prog_T20L30_4-ft-top-10", 10)
optimize("prog_T20L30_5-ft+5k-top-10", 10)

optimize("naive_T30L30_4-ft-top-10", 10)
optimize("normal_T30L30_1-ft-top-10", 10)
optimize("prog_T30L30_1-ft-top-10", 10)
optimize("prog_T30L30_1-ft+5k-top-10", 10)