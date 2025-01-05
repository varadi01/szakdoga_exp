from impl.rule_based import RuleBasedPlayer
from impl.scenario import SimpleGame, Environment, TileState
from random import randint
import os


def generate_data(filename: str = "dataset", data_length: int = 1000):
    f = open(os.path.join("..", "res", filename + ".txt"), "+w")
    alg = RuleBasedPlayer(SimpleGame()) # just need the guy itself
    for _ in range(data_length):
        env = Environment(
            TileState(randint(0, 2)),
            TileState(randint(0, 2)),
            TileState(randint(0, 2)),
            TileState(randint(0, 2))
        )
        action = alg.act(env)
        line = f"{env.up.value},{env.right.value},{env.down.value},{env.left.value};{action.value}"
        f.write(line)
        if _ < data_length-1:
            f.write("\n")
    f.close()


generate_data("ge_dataset", 500)