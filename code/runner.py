#Do I make a very basic rule based one as well for comparison?

#What DO we need to store?
# - Game data, basic: number of steps achieved, trees consumed, cause of death, Game parameters(tree ratio, lion ratio, starting food, food gain), food at death?
# - evolutionary: species(steps to offspring, food to offspring, forget rate, ),
# - deepl:
# - rein:

#Data model
#


#TODO sometimes we dont lose food? step translation faulty
from scenario import Game, ResultOfStep
from rule_based import RuleBasedPlayer


def run(alg):
    steps_taken = 0
    while True:
        result, step = alg.act()
        steps_taken += 1

        if result in (ResultOfStep.STARVED, ResultOfStep.ENCOUNTERED_LION): #might need tree
            print(f"steps taken: {steps_taken}, cause of death: {result}")
            break


run(RuleBasedPlayer(Game()))