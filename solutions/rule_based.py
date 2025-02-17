from game_environment.scenario import SimpleGame, ExtendedGame, Step, ExtendedStep,  Environment, TileState, ResultOfStep, ExtendedResultOfStep
from random import choice
from utils.db_context import get_instance
from utils.db_entities import RecordModel

class RuleBasedPlayer:
    def __init__(self, given_scenario = None, tree_ratio: float = None, lion_ratio: float = None,):
        if given_scenario is not None:
            self.game = given_scenario
        else:
            if tree_ratio is not None:
                self.game = SimpleGame(tree_ratio=tree_ratio, lion_ratio=lion_ratio)
            else:
                self.game = SimpleGame()

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
                game_result = RecordModel.GameResult.EATEN_BY_LION
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

class ExtendedRuleBasePlayer:
    """non-pursuit environments"""

    def __init__(self, given_scenario=None, tree_ratio: float = None, lion_ratio: float = None, ):
        if given_scenario is not None:
            self.game = given_scenario
        else:
            if tree_ratio is not None:
                self.game = ExtendedGame(tree_ratio=tree_ratio, lion_ratio=lion_ratio)
            else:
                self.game = ExtendedGame()

    def act(self, env: Environment = None) -> Step:
        """env only used for ease of data generation"""
        # get env
        if env is None:
            environment = self.game.get_environment()
        else:
            environment = env
        # decide step
        trees = []
        land = []
        lions = []

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

        if environment.up == TileState.LION:
            lions.append(Step.UP)
        if environment.right == TileState.LION:
            lions.append(Step.RIGHT)
        if environment.down == TileState.LION:
            lions.append(Step.DOWN)
        if environment.left == TileState.LION:
            lions.append(Step.LEFT)

        if len(lions) > 1:
            c = Step.STAY
        elif len(lions) == 1:
            c = lions[0]
        elif len(trees) > 0:
            c = choice(trees)
        elif len(land) > 0:
            c = choice(land)
        return c

    def eval(self):
        steps_taken = 0
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
                game_result = RecordModel.GameResult.EATEN_BY_LION
                break

            if self.game.is_won():
                print(f"steps taken: {steps_taken}, won the game")
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
            "extended",
            RecordModel.make_parameter_string(self.game.TREE_RATIO, self.game.LION_RATIO)
        )
        db.insert(record)

# RuleBasedPlayer(tree_ratio=0.4, lion_ratio=0.4).eval()
