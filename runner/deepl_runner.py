from solutions.deepl import SmallClassDeepl
import os

PATH_TO_SIMPLE_GENERATED_LEARNING_DATASET = os.path.join("..", "res", "gt_dataset.txt")
PATH_TO_SIMPLE_GENERATED_EVALUATION_DATASET = os.path.join("..", "res", "ge_dataset.txt")


def main():
    da = SmallClassDeepl()
    da.describe()
    da.learn(PATH_TO_SIMPLE_GENERATED_LEARNING_DATASET)
    da.evaluate(PATH_TO_SIMPLE_GENERATED_EVALUATION_DATASET)


if __name__ == '__main__':
    main()