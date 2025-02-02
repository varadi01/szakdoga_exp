from game_environment.scenario import SimpleGame, Step, Environment, TileState, ResultOfStep
from random import choice
from utils.db_context import get_instance
from utils.db_entities import RecordModel

class RuleBasedPlayer:
    def __init__(self, game = SimpleGame):
        self.game = game()

    def act(self, env: Environment = None) -> Step:
        """env only used for ease of data generation"""
        #get env
        if env is None:
            environment = self.game.get_environment()
        else:
            environment = env
        #decide step
        trees = []
        land = []

        if environment.up == TileState.TREE:
            trees.append(Step.UP)
        if environment.right == TileState.TREE:
            trees.append(Step.RIGHT)
        if environment.down == TileState.TREE:
            trees.append(Step.DOWN)
        if environment.left == TileState.TREE:
            trees.append(Step.LEFT)

        if environment.up == TileState.LAND:
            land.append(Step.UP)
        if environment.right == TileState.LAND:
            land.append(Step.RIGHT)
        if environment.down == TileState.LAND:
            land.append(Step.DOWN)
        if environment.left == TileState.LAND:
            land.append(Step.LEFT)

        #make step
        if len(trees) > 0:
            c =choice(trees)
            return c
        if len(land) > 0:
            cl = choice(land)
            return cl
        return choice((Step.RIGHT, Step.LEFT, Step.UP, Step.DOWN))

    def eval(self):
        #only runs once, make it a proper evaluation
        steps_taken = 0
        game_result = None
        while True:
            step = self.act()
            steps_taken += 1
            result = self.game.make_step(step)

            if result == ResultOfStep.STARVED:  # might need tree
                print(f"steps taken: {steps_taken}, cause of death: {result}")
                game_result = RecordModel.GameResult.STARVED
                break

            if result == ResultOfStep.EATEN_BY_LION:  # might need tree
                print(f"steps taken: {steps_taken}, cause of death: {result}")
                game_result = RecordModel.GameResult.LION
                break

            if steps_taken > 500:
                print(f"steps taken: {steps_taken}, survived over 500 steps")
                game_result = RecordModel.GameResult.COMPLETE
                break

        self.save_record(steps_taken, game_result)


    def save_record(self, steps, result):
        db = get_instance("rule_based_records", 'r')
        record = RecordModel(
            "rule_based",
            result,
            steps,
            self.game.steps_left,
            "simple",
            RecordModel.make_parameter_string(self.game.TREE_RATIO, self.game.LION_RATIO)
        )
        db.insert(record)

