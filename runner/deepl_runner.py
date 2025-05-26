from solutions.deepl import Deepl, ExtendedDeepl, save_model, load_model, MultiLabelDeepl, MultiLabelDeeplExtended, BiggerDeepl
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

    #todo try bigger network



    # label = np.zeros(5,) #!!! CHANGE BACK-FORTH
    # da_ex = ExtendedDeepl()
    # da_ex.describe()
    # da_ex.learn(PATH_TO_EXTENDED_GENERATED_LEARNING_DATASET, epochs=1500, batch=100)
    # save_model(da_ex.model, "mlp_extended_very-long")
    #
    #
    # for i in range(1):
    #     da_ex.test(ExtendedGame(tree_ratio=0.3, lion_ratio=.3))

    # ml_deepl = MultiLabelDeepl()
    # ml_deepl.learn(PATH_TO_SIMPLE_MULTILABEL_GENERATED_LEARNING_DATASET, batch=100, epochs=1500)
    # save_model(ml_deepl.model, "mlp_simple_multilabel_very-long")
    #
    # for i in range(1):
    #     ml_deepl.test(SimpleGame())


    # ml_deepl_ex = MultiLabelDeeplExtended()
    # ml_deepl_ex.learn(PATH_TO_EXTENDED_MULTILABEL_GENERATED_LEARNING_DATASET, batch=100, epochs=1500)
    # save_model(ml_deepl_ex.model, "mlp_extended_multilabel_very-long")
    #
    # for i in range(1):
    #     ml_deepl_ex.test(ExtendedGame())


    # isolated test ----------------------------------------------------------------

    # deepl_simple = Deepl()
    # deepl_simple.learn(PATH_TO_SIMPLE_GENERATED_LEARNING_DATASET, batch=100, epochs=500)
    #
    # # for _ in range(10):
    # #     deepl_simple.test(SimpleGame(lion_ratio=0.3, tree_ratio=0.3))
    #
    # save_model(deepl_simple.model, "deepl_simple_long_3")

    # ----

    # deepl_simple_ml = MultiLabelDeepl()
    # deepl_simple_ml.learn(PATH_TO_SIMPLE_MULTILABEL_GENERATED_LEARNING_DATASET, batch=100, epochs=20)

    # save_model(deepl_simple_ml.model, "MLM_short_4")
    #
    # # ----
    #
    # deepl_simple_bigger = BiggerDeepl()
    # deepl_simple_bigger.learn(PATH_TO_SIMPLE_GENERATED_LEARNING_DATASET, batch=100, epochs=500)
    #
    # save_model(deepl_simple_bigger.model, "deepl_simple_big_long")


    # deepl_ex = ExtendedDeepl()
    # deepl_ex.learn(PATH_TO_EXTENDED_GENERATED_LEARNING_DATASET, epochs=500)
    #
    # save_model(deepl_ex.model, "deepl_ex_long_3")


    # deepl_ex_ml = MultiLabelDeeplExtended()
    # deepl_ex_ml.learn(PATH_TO_EXTENDED_MULTILABEL_GENERATED_LEARNING_DATASET, epochs=500)
    #
    # save_model(deepl_ex_ml.model, "MLM_ex_long_3")

    # for i in range(1,11):
    #     deepl = Deepl()
    #     deepl.learn(PATH_TO_SIMPLE_GENERATED_LEARNING_DATASET, batch=100, epochs=100)
    #     save_model(deepl.model, f"SLM_medium_{i}")

    # for i in range(1,11):
    #     deepl = MultiLabelDeepl()
    #     deepl.learn(PATH_TO_SIMPLE_MULTILABEL_GENERATED_LEARNING_DATASET, batch=100, epochs=500)
    #     save_model(deepl.model, f"MLM_long_{i}")

    # for i in range(1,6):
    #     dl = ExtendedDeepl()
    #     dl.learn(PATH_TO_EXTENDED_GENERATED_LEARNING_DATASET, batch=100, epochs=500)
    #     save_model(dl.model, f"SLM_ex_long_{i}")


    # for i in range(1,6):
    #     dl = MultiLabelDeeplExtended()
    #     dl.learn(PATH_TO_EXTENDED_MULTILABEL_GENERATED_LEARNING_DATASET, batch=100, epochs=500)
    #     save_model(dl.model, f"MLM_ex_long_{i}")

    # end isolated test ----------------------------------------------------------------


    pass


if __name__ == '__main__':
    main()