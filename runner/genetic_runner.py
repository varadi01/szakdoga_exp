from solutions.genetic import Genetic, GeneticNaive, Individual, ExtendedIndividual, models_to_individuals
from game_environment.scenario import ExtendedGame
from utils.db_context import get_instance


#naive progressive makes no sense
#we can do progressive with normal Genetic, mixing action_sets and forgetting actions

#inheriting food? interesting point

#TODO ex progressive needs initialize_gen to be set to ExtendedGame

#todo train further -> performance based selection
#rn they cant do 30-50

#naive simple for 300 cycles
#normal simple 200 cycles, longer is 200 mistake
#naive extended is 100


def main():

    # naive_simple = GeneticNaive(100000, game_tree_ratio=0.15, game_lion_ratio=0.4, individual_type=Individual)
    # naive_simple.train(300, True, "naive_simple_T15L40")


    # normal_simple = Genetic(1000, game_tree_ratio=0.10, game_lion_ratio=0.40, individual_type=Individual)
    # normal_simple.train(200, True, "normal_simple_T10L40")

    #very hard from 30,30 we need a LOT of individuals (around 10k)
    #interesting U curve


    # naive_extended = GeneticNaive(10000, game_tree_ratio=0.50, game_lion_ratio=0.40, individual_type=ExtendedIndividual)
    # naive_extended.train(300, True, "naive_extended_T50L40")


    # normal_extended = Genetic(1000, game_tree_ratio=0.10, game_lion_ratio=0.4, individual_type=ExtendedIndividual)
    # normal_extended.train(200, True, "normal_extended_T10L40")


    # db = get_instance("progressive_simple_T10L40", 'g')
    # models = db.get_all()
    # normal_simple_progressive = Genetic(1000, game_tree_ratio=0.10, game_lion_ratio=0.55,
    #                                     individual_type=Individual, existing_generation=models_to_individuals(models))
    # normal_simple_progressive.train(100, True, "progressive_simple_T10L55")


    pass


if __name__ == '__main__':
    main()