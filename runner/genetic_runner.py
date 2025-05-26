from solutions.genetic import Genetic, GeneticNaive, Individual, ExtendedIndividual, models_to_individuals
from game_environment.scenario import ExtendedGame
from utils.db_context import get_instance


#inheriting food? interesting point



# train further -> performance based selection
#rn they cant do 30-50

# 0 - T30L10
# 1 - T30L30
# 2 - T30L40
# 3 - T30L50
# 4 - T20L40
# 5 - T15L40
# 6 - T10L40

#cycles
# short - 100
# normal - 300
# long - 500
# very_long - 1000 #this took 35 minutes :(
# longest - 1500 (limited pop, 500 is fairly quick)
# more longest - 5000 (limited pop, 500 is fairly quick)

# pop
# nothing - 100
# medium - 1000
# large - 10k

def main():
    #gen isolated test ---------------------------------

    # db = get_instance("naive_simple_1_7k", 'g', "simple_genetic_ind")
    # models = db.get_all()
    # naive_simple = GeneticNaive(500, game_tree_ratio=0.3, game_lion_ratio=0.3,
    #                             individual_type=Individual)#, existing_generation=models_to_individuals(models))
    # naive_simple.train(10000, True, "naive_simple_1_10k")
    #second round did 500 more steps, shows how unlikely it is that they learn
    #

    # db = get_instance("normal_simple_4_prog_long", 'g', "simple_genetic_ind")
    # models = db.get_all()
    # normal_simple = Genetic(500, game_tree_ratio=0.2, game_lion_ratio=0.4,
    #                             individual_type=Individual, existing_generation=models_to_individuals(models))
    # normal_simple.train(500, True, "normal_simple_4_prog_long-cont")

    # end gen isolated test ----------------------------

    #second gen isolated test --------------------------------------------

    #1 40-30
    #2 40-40
    #3 30-30
    #4 20-40

    # naive_simple = GeneticNaive(500, game_tree_ratio=0.4, game_lion_ratio=0.3, individual_type=Individual)
    # naive_simple.train(10000, True, "naive_simple_T40L30_10k_1")

    # naive_simple = GeneticNaive(500, game_tree_ratio=0.4, game_lion_ratio=0.4, individual_type=Individual)
    # naive_simple.train(10000, True, "naive_simple_T40L30_10k_2") #todo error

    # naive_simple = GeneticNaive(500, game_tree_ratio=0.3, game_lion_ratio=0.3, individual_type=Individual)
    # naive_simple.train(5000, True, "naive_simple_T30L30_5k_3")

    # naive_simple = GeneticNaive(500, game_tree_ratio=0.2, game_lion_ratio=0.4, individual_type=Individual)
    # naive_simple.train(10000, True, "naive_simple_T20L40_10k_4")

    # for i in range(1,6):
        # naive_simple_numbered = GeneticNaive(500, game_tree_ratio=0.2, game_lion_ratio=0.3, individual_type=Individual)
        # naive_simple_numbered.train(10000, True, f"naive_T20L30_{i}")
        #
        # normal_simple_numbered = Genetic(500, game_tree_ratio=0.2, game_lion_ratio=0.3, individual_type=Individual)
        # normal_simple_numbered.train(500, True, f"normal_T20L30_{i}")

        # normal_simple_numbered_hard = Genetic(500, game_tree_ratio=0.15, game_lion_ratio=0.4, individual_type=Individual)
        # normal_simple_numbered_hard.train(500, True, f"normal_T15L40_{i}")
        #
        # normal_simple_numbered1k = Genetic(500, game_tree_ratio=0.3, game_lion_ratio=0.3, individual_type=Individual)
        # normal_simple_numbered1k.train(1000, True, f"normal_T30L30_1k_{i}")
        #
        # prog_simple_numbered = Genetic(500, game_tree_ratio=0.2, game_lion_ratio=0.3, individual_type=Individual)
        # prog_simple_numbered.train(500, True, f"prog_T20L30_{i}")

        # db = get_instance(f"prog_T20L30_{i}-ft", 'g', "extended_genetic_ind")
        # models = db.get_all()
        # prog_simple = Genetic(500, game_tree_ratio=0.2, game_lion_ratio=0.3, individual_type=Individual, existing_generation=models_to_individuals(models))
        # prog_simple.train(5000, True, f"prog_T20L30_{i}-ft+5k")

        #more progressive
        # db = get_instance(f"prog_T20L30_{i}-ft+5k", 'g', "extended_genetic_ind")
        # models = db.get_all()
        # prog_simple = Genetic(500, game_tree_ratio=0.15, game_lion_ratio=0.3, individual_type=Individual, existing_generation=models_to_individuals(models))
        # prog_simple.train(2000, True, f"prog_T15L30_{i}-ft+5k_T15L30+2k")
        # pass


    # normal_simple = Genetic(500, game_tree_ratio=0.4, game_lion_ratio=0.3, individual_type=Individual)
    # normal_simple.train(500, True, "normal_simple_T40L30_5h_long")
    #
    # normal_simple = Genetic(500, game_tree_ratio=0.4, game_lion_ratio=0.4, individual_type=Individual)
    # normal_simple.train(500, True, "normal_simple_T40L40_5h_long_2")
    #
    # normal_simple = Genetic(500, game_tree_ratio=0.3, game_lion_ratio=0.3, individual_type=Individual)
    # normal_simple.train(500, True, "normal_simple_T30L30_5h_long")
    #
    # normal_simple = Genetic(500, game_tree_ratio=0.2, game_lion_ratio=0.4, individual_type=Individual)
    # normal_simple.train(500, True, "normal_simple_T20L40_5h_long")

    # db = get_instance("prog_simple_T40L30_5h_long-ft", 'g', "simple_genetic_ind_2")
    # models = db.get_all()
    # prog_simple = Genetic(500, game_tree_ratio=0.4, game_lion_ratio=0.3, individual_type=Individual, existing_generation=models_to_individuals(models))
    # prog_simple.train(5000, True, "prog_simple_T40L30_5h_long+5k")
    #
    # db = get_instance("prog_simple_T40L40_5h_long-ft", 'g', "simple_genetic_ind_2")
    # models = db.get_all()
    # prog_simple = Genetic(500, game_tree_ratio=0.4, game_lion_ratio=0.4, individual_type=Individual, existing_generation=models_to_individuals(models))
    # prog_simple.train(5000, True, "prog_simple_T40L40_5h_long+5k")
    #
    # db = get_instance("prog_simple_T30L30_5h_long-ft", 'g', "simple_genetic_ind_2")
    # models = db.get_all()
    # prog_simple = Genetic(500, game_tree_ratio=0.3, game_lion_ratio=0.3, individual_type=Individual, existing_generation=models_to_individuals(models))
    # prog_simple.train(5000, True, "prog_simple_T30L30_5h_long+5k")

    # db = get_instance("prog_simple_T20L40_5h_long-ft", 'g', "simple_genetic_ind_2")
    # models = db.get_all()
    # prog_simple = Genetic(500, game_tree_ratio=0.2, game_lion_ratio=0.4, individual_type=Individual, existing_generation=models_to_individuals(models))
    # prog_simple.train(10000, True, "prog_simple_T20L40_5h_long+10k")



    # exxx


    # naive_ex = GeneticNaive(500, game_tree_ratio=0.3, game_lion_ratio=0.3, individual_type=ExtendedIndividual)
    # naive_ex.train(5000, do_save=True, target_collection="naive_ex_T30L30_5k")

    # naive_ex = GeneticNaive(500, game_tree_ratio=0.4, game_lion_ratio=0.2, individual_type=ExtendedIndividual)
    # naive_ex.train(5000, do_save=True, target_collection="naive_ex_T40L20_5k")

    # naive_ex = GeneticNaive(100, game_tree_ratio=0.3, game_lion_ratio=0.3, individual_type=ExtendedIndividual)
    # naive_ex.train(10000, do_save=True, target_collection="naive_ex_T30L30_10k_1h_3")
    #
    # naive_ex = GeneticNaive(500, game_tree_ratio=0.3, game_lion_ratio=0.4, individual_type=ExtendedIndividual)
    # naive_ex.train(10000, do_save=True, target_collection="naive_ex_T30L40_10k")
    #
    # naive_ex = GeneticNaive(500, game_tree_ratio=0.2, game_lion_ratio=0.4, individual_type=ExtendedIndividual)
    # naive_ex.train(5000, do_save=True, target_collection="naive_ex_T20L40_5k")
    #
    # naive_ex = GeneticNaive(500, game_tree_ratio=0.4, game_lion_ratio=0.3, individual_type=ExtendedIndividual)
    # naive_ex.train(10000, do_save=True, target_collection="naive_ex_T40L30_10k_2")

    # normal_ex = Genetic(500, game_tree_ratio=0.3, game_lion_ratio=0.3, individual_type=ExtendedIndividual)
    # normal_ex.train(500, True, "normal_ex_T30L30_5h_long")
    #
    # normal_ex = Genetic(500, game_tree_ratio=0.4, game_lion_ratio=0.3, individual_type=ExtendedIndividual)
    # normal_ex.train(500, True, "normal_ex_T40L30_5h_long")
    #
    # normal_ex = Genetic(500, game_tree_ratio=0.3, game_lion_ratio=0.4, individual_type=ExtendedIndividual)
    # normal_ex.train(500, True, "normal_ex_T30L40_5h_long")
    #
    # normal_ex = Genetic(500, game_tree_ratio=0.4, game_lion_ratio=0.2, individual_type=ExtendedIndividual)
    # normal_ex.train(500, True, "normal_ex_T40L20_5h_long")


    # models = db.get_all()
    # prog_ex = Genetic(500, game_tree_ratio=0.3, game_lion_ratio=0.3, individual_type=ExtendedIndividual, existing_generation=models_to_individuals(models))
    # prog_ex.train(5000, True, "prog_ex_T30L30_5h_long+5k")
    #
    # db = get_instance("prog_ex_T40L30_5h_long-ft", 'g', "extended_genetic_ind_prog")
    # models = db.get_all()
    # prog_ex = Genetic(500, game_tree_ratio=0.4, game_lion_ratio=0.3, individual_type=ExtendedIndividual, existing_generation=models_to_individuals(models))
    # prog_ex.train(5000, True, "prog_ex_T40L30_5h_long+5k")
    #
    # db = get_instance("prog_ex_T30L40_5h_long-ft", 'g', "extended_genetic_ind_prog")
    # models = db.get_all()
    # prog_ex = Genetic(500, game_tree_ratio=0.3, game_lion_ratio=0.4, individual_type=ExtendedIndividual, existing_generation=models_to_individuals(models))
    # prog_ex.train(5000, True, "prog_ex_T30L40_5h_long+5k")
    #
    # db = get_instance("prog_ex_T40L20_5h_long-ft", 'g', "extended_genetic_ind_prog")
    # models = db.get_all()
    # prog_ex = Genetic(500, game_tree_ratio=0.4, game_lion_ratio=0.2, individual_type=ExtendedIndividual, existing_generation=models_to_individuals(models))
    # prog_ex.train(5000, True, "prog_ex_T40L20_5h_long+5k")




    # for i in range(12,13):
    #     naive_ex = GeneticNaive(500, game_tree_ratio=0.3, game_lion_ratio=0.3, individual_type=ExtendedIndividual)
    #     naive_ex.train(10000, do_save=True, target_collection=f"naive_ex_T30L30_10k_{i}")
        #
        # normal_ex = Genetic(500, game_tree_ratio=0.3, game_lion_ratio=0.2, individual_type=ExtendedIndividual)
        # normal_ex.train(500, True, f"normal_ex_T30L20_{i}")
        #
        # prog_ex = Genetic(500, game_tree_ratio=0.3, game_lion_ratio=0.2, individual_type=ExtendedIndividual) #, existing_generation=models_to_individuals(models))
        # prog_ex.train(5000, True, f"prog_ex_T30L20_{i}")

        # prog_ex_real = Genetic(500, game_tree_ratio=0.3, game_lion_ratio=0.2, individual_type=ExtendedIndividual)
        # prog_ex_real.train(500, True, f"prog_ex_T30L20_{i}_real")

        # db = get_instance(f"prog_ex_T30L20_{i}_real-ft", 'g', "extended_genetic_ind_actual")
        # models = db.get_all()
        # prog_ex_p = Genetic(500, game_tree_ratio=0.3, game_lion_ratio=0.2, individual_type=ExtendedIndividual, existing_generation=models_to_individuals(models))
        # prog_ex_p.train(5000, True, f"prog_ex_T30L20_{i}_real-ft+5k")
    #end second gen isolated test --------------------------------------------

    # normal_simple = Genetic(1000, game_tree_ratio=0.1, game_lion_ratio=0.4, individual_type=Individual)
    # normal_simple.train(300, True, "normal_simple_6_medium")

    #interesting U curve
    #the ones I have is the least amount of pop required
    #gets real hard after 30-30
    # 20-40 actually too hard for naive


    # naive_extended = GeneticNaive(10000, game_tree_ratio=0.2, game_lion_ratio=0.4, individual_type=ExtendedIndividual)
    # naive_extended.train(300, True, "naive_extended_4_large")


    # normal_extended = Genetic(1000, game_tree_ratio=0.1, game_lion_ratio=0.4, individual_type=ExtendedIndividual)
    # normal_extended.train(300, True, "normal_extended_6_medium")


    # db = get_instance("progressive_simple_6_medium_4", 'g')
    # models = db.get_all()
    # normal_simple_progressive = Genetic(100, game_tree_ratio=0.1, game_lion_ratio=0.4,
    #                                     individual_type=Individual, existing_generation=models_to_individuals(models))
    # normal_simple_progressive.train(100, True, "progressive_simple_6_medium_5")


    # db = get_instance("progressive_extended_5_2", 'g')
    # models = db.get_all()
    # normal_extended_progressive = Genetic(100, game_tree_ratio=0.1, game_lion_ratio=0.4,
    #                                       individual_type=ExtendedIndividual, existing_generation=models_to_individuals(models))
    # normal_extended_progressive.train(100, True, "progressive_extended_6_2")


    n = GeneticNaive(100, game_tree_ratio=0.15, game_lion_ratio=0.4, individual_type=Individual)
    n.train(100,False,"")
    pass


if __name__ == '__main__':
    main()