#Do I make a very basic rule based one as well for comparison?
from impl.genetic import Genetic
#What DO we need to store?
# - Game data, basic: number of steps achieved, trees consumed, cause of death, Game parameters(tree ratio, lion ratio, starting food, food gain), food at death?
# - evolutionary: species(steps to offspring, food to offspring, forget rate, ),
# - deepl:
# - rein:

#Data model
#


from scenario import Game, ResultOfStep
from rule_based import RuleBasedPlayer

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
    g = Genetic(Game)
    g.run()

    pass



if __name__ == "__main__":
    main()