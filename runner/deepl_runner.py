from solutions.deepl import Deepl, ExtendedDeepl, save_model, load_model
from game_environment.scenario import SimpleGame, ExtendedGame
import os

PATH_TO_SIMPLE_GENERATED_LEARNING_DATASET = os.path.join("..", "res", "gt_dataset.txt")
PATH_TO_SIMPLE_GENERATED_EVALUATION_DATASET = os.path.join("..", "res", "ge_dataset.txt")
PATH_TO_EXTENDED_GENERATE_LEARNING_DATASET = os.path.join("..", "res", "gt_ex_dataset.txt")
PATH_TO_EXTENDED_GENERATE_EVALUATION_DATASET = os.path.join("..", "res", "ge_ex_dataset.txt")

#!!!
#learns quickly and well, but we have a fairly high learning rate
#!!!

def main():
    # da = Deepl()
    # da.describe()
    # da.learn(PATH_TO_SIMPLE_GENERATED_LEARNING_DATASET, epochs=20, batch=100, validation_split=0.05)
    # save_model(da.model, "simple_mlp_2_short-train")
    #
    # # da.evaluate(PATH_TO_SIMPLE_GENERATED_EVALUATION_DATASET)
    # # da.test(SimpleGame(tree_ratio=0.5, lion_ratio=0.3))
    #
    # for i in range(10):
    #     da.test(SimpleGame())

    #todo
    # label = np.zeros(5,) #!!! CHANGE BACK-FORTH
    da_ex = ExtendedDeepl()
    da_ex.describe()
    da_ex.learn(PATH_TO_EXTENDED_GENERATE_LEARNING_DATASET, epochs=10, batch=100)

    for i in range(10):
        da_ex.test(ExtendedGame())


if __name__ == '__main__':
    main()