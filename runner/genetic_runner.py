from solutions.genetic import Genetic, GeneticNaive, Individual, ExtendedIndividual, models_to_individuals
from utils.db_context import get_instance


#naive progressive makes no sense
#we can do progressive with normal Genetic, mixing action_sets and forgetting actions

#TODO ex progressive needs initialize_gen to be set to ExtendedGame

#todo train further -> performance based selection
#rn they cant do 30-50

def main():
    #"genetic_naive_abundant"
    # g = Genetic()
    # g.run()

    # gn = GeneticNaive(1000)
    # gn.train(50)

    # db = get_instance("gen_prog_normal_simple_T20L50",'g')
    # existing_gen = models_to_individuals(db.get_all())

    #
    # naive_simple = GeneticNaive(1000, game_tree_ratio=0.2, game_lion_ratio=0.3, individual_type=Individual)
    # naive_simple.train(300, False, "gen_naive_simple_T30L50")

    # normal_simple = Genetic(1000, game_tree_ratio=0.2, game_lion_ratio=0.7, individual_type=Individual, existing_generation=existing_gen)
    # normal_simple.train(100, True, "gen_prog_normal_simple_T20L70")


    #looks like its muuch harder
    # naive_extended = GeneticNaive(100000, game_tree_ratio=0.3, game_lion_ratio=0.3, individual_type=ExtendedIndividual)
    # naive_extended.train(70, True, "gen_naive_extended_partial_T30L30")

    normal_extended = Genetic(1000, game_tree_ratio=0.3, game_lion_ratio=0.4, individual_type=ExtendedIndividual)
    normal_extended.train(300, True, "gen_normal_extended_T30L40")

    #todo extended progressive
    #todo consider -ft in normal
    pass


if __name__ == '__main__':
    main()