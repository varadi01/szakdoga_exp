#Do I make a very basic rule based one as well for comparison?

#What DO we need to store?
# - Game data, basic: number of steps achieved, trees consumed, cause of death, Game parameters(tree ratio, lion ratio, starting food, food gain), food at death?
# - evolutionary: species(steps to offspring, food to offspring, forget rate, ),
# - deepl:
# - rein:

#Data model
#

import tensorflow as tf
import os

from impl.deepl import SmallClassDeepl
from scenario import SimpleGame, ResultOfStep
from stable_baselines3 import PPO, A2C

from rule_based import RuleBasedPlayer
from genetic import Genetic, GeneticNaive
from rl import Agent

PATH_TO_SIMPLE_GENERATED_LEARNING_DATASET = os.path.join("..", "res", "gt_dataset.txt")
PATH_TO_SIMPLE_GENERATED_EVALUATION_DATASET = os.path.join("..", "res", "ge_dataset.txt")


def main():


    #run(RuleBasedPlayer(SimpleGame()))

    # g = Genetic()
    # g.run()

    gn = GeneticNaive(1000)
    gn.train(250)

    #TODO try different algs, tweak learning rate, rewards, explore/exploit?

    # agent = Agent(A2C, "test", False)
    # agent.learn(1000)
    # #agent.evaluate(10)
    # agent.test()


    # da = SmallClassDeepl()
    # da.learn(PATH_TO_SIMPLE_GENERATED_LEARNING_DATASET)
    pass



if __name__ == "__main__":
    main()