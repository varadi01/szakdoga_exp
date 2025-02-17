from solutions.deepl import Deepl, ExtendedDeepl, save_model, load_model, MultiLabelDeepl, MultiLabelDeeplExtended
from game_environment.scenario import SimpleGame, ExtendedGame
import os

PATH_TO_SIMPLE_GENERATED_LEARNING_DATASET = os.path.join("..", "res", "gt_dataset.txt")
PATH_TO_SIMPLE_GENERATED_EVALUATION_DATASET = os.path.join("..", "res", "ge_dataset.txt")
PATH_TO_SIMPLE_MULTILABEL_GENERATED_LEARNING_DATASET = os.path.join("..", "res", "gt_ml_dataset.txt")
PATH_TO_EXTENDED_GENERATED_LEARNING_DATASET = os.path.join("..", "res", "gt_ex_dataset.txt")
PATH_TO_EXTENDED_MULTILABEL_GENERATED_LEARNING_DATASET = os.path.join("..", "res", "gt_ex_ml_dataset.txt")
PATH_TO_EXTENDED_GENERATE_EVALUATION_DATASET = os.path.join("..", "res", "ge_ex_dataset.txt")

#!!!
#learns quickly and well, but we have a fairly high learning rate
#!!!

# compare time it takes to decide step

# lr
# normal .02

# train length
# short - 20 epochs
# medium - 100 epochs
# long - 500 epochs
# very-long - 1500 epochs

# cant do extended, due to neural network structure
# I'd need to do a no attack dataset

def main():
    # da = Deepl()
    # da.describe()
    # da.learn(PATH_TO_SIMPLE_GENERATED_LEARNING_DATASET, epochs=1500, batch=100, validation_split=0.05)
    # save_model(da.model, "mlp_simple_very-long")
    #
    # # da.evaluate(PATH_TO_SIMPLE_GENERATED_EVALUATION_DATASET)
    # # da.test(SimpleGame(tree_ratio=0.5, lion_ratio=0.3))
    #
    # for i in range(10):
    #     da.test(SimpleGame())

    # label = np.zeros(5,) #!!! CHANGE BACK-FORTH
    # da_ex = ExtendedDeepl()
    # da_ex.describe()
    # da_ex.learn(PATH_TO_EXTENDED_GENERATE_LEARNING_DATASET, epochs=1500, batch=100)
    # save_model(da_ex.model, "mlp_extended_very-long")
    #
    #
    # for i in range(10):
    #     da_ex.test(ExtendedGame(tree_ratio=0.4, lion_ratio=.1))

    # ml_deepl = MultiLabelDeepl()
    # ml_deepl.learn(PATH_TO_SIMPLE_MULTILABEL_GENERATED_LEARNING_DATASET, batch=100, epochs=1500)
    # save_model(ml_deepl.model, "mlp_simple_multilabel_very-long")
    #
    # for i in range(1):
    #     ml_deepl.test(SimpleGame())

    ml_deepl_ex = MultiLabelDeeplExtended()
    ml_deepl_ex.learn(PATH_TO_EXTENDED_MULTILABEL_GENERATED_LEARNING_DATASET, batch=100, epochs=1500)
    save_model(ml_deepl_ex.model, "mlp_extended_multilabel_very-long")

    for i in range(5):
        ml_deepl_ex.test(ExtendedGame())


if __name__ == '__main__':
    main()