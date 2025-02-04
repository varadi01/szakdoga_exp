from solutions.deepl import Deepl
from game_environment.scenario import SimpleGame
import os

PATH_TO_SIMPLE_GENERATED_LEARNING_DATASET = os.path.join("..", "res", "gt_dataset.txt")
PATH_TO_SIMPLE_GENERATED_EVALUATION_DATASET = os.path.join("..", "res", "ge_dataset.txt")


def main():
    da = Deepl()
    da.describe()
    da.learn(PATH_TO_SIMPLE_GENERATED_LEARNING_DATASET)
    da.evaluate(PATH_TO_SIMPLE_GENERATED_EVALUATION_DATASET)
    da.test(SimpleGame(tree_ratio=0.5, lion_ratio=0.3))


if __name__ == '__main__':
    main()