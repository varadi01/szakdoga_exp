#Do I make a very basic rule based one as well for comparison?

#What DO we need to store?
# - Game data, basic: number of steps achieved, trees consumed, cause of death, Game parameters(tree ratio, lion ratio, starting food, food gain), food at death?
# - evolutionary: species(steps to offspring, food to offspring, forget rate, ),
# - deepl:
# - rein:

#Data model
#

import os

from solutions.deepl import SmallClassDeepl

from solutions.rule_based import RuleBasedPlayer

from solutions.genetic import GeneticNaive, Genetic

from utils.db_context import get_instance, RecordSerializer




def main():

    # for _ in range(10):
    #     RuleBasedPlayer().eval()

    # db = get_instance("rule_based_records")
    # records = db.get_all()
    # serializer = RecordSerializer()
    # for doc in records:
    #     print(serializer.deserialize(doc))


    # agent = Agent(A2C, "test", False)
    # agent.learn(1000)
    # agent.evaluate(10)
    # agent.test()



    pass



if __name__ == "__main__":
    main()