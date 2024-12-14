#Do I make a very basic rule based one as well for comparison?

#What DO we need to store?
# - Game data, basic: number of steps achieved, trees consumed, cause of death, Game parameters(tree ratio, lion ratio, starting food, food gain), food at death?
# - evolutionary: species(steps to offspring, food to offspring, forget rate, ),
# - deepl:
# - rein:

#Data model
#


from scenario import Game, ResultOfStep
from stable_baselines3 import PPO, A2C

from rule_based import RuleBasedPlayer
from genetic import Genetic
from rl import Agent

#TODO put into rule based
def run(alg):
    steps_taken = 0
    while True:
        result, step = alg.act()
        steps_taken += 1

        if result in (ResultOfStep.STARVED, ResultOfStep.ENCOUNTERED_LION): #might need tree
            print(f"steps taken: {steps_taken}, cause of death: {result}")
            break

        if steps_taken > 500:
            print(f"steps taken: {steps_taken}, survived over 500 steps")
            break




def main():
    #run(RuleBasedPlayer(Game()))

    g = Genetic()
    g.run()

    #TODO try different algs, tweak learning rate, rewards, explore/exploit?
    # agent = Agent(A2C, "test", False)
    # agent.learn(10000)
    # agent.test()
    pass



if __name__ == "__main__":
    main()