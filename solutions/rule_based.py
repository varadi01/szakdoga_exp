from impl.scenario import SimpleGame, Step, Environment, TileState, ResultOfStep
from random import choice

class RuleBasedPlayer:
    def __init__(self, game: SimpleGame):
        self.game = game

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
            print(f"step {c}")
            return c
        if len(land) > 0: #should always be true?
            cl = choice(land)
            print(f"step {cl}")
            return cl
        return choice((Step.RIGHT, Step.LEFT, Step.UP, Step.DOWN))

    def eval(self):
        #only runs once, make it a proper evaluation
        steps_taken = 0
        while True:
            result, step = self.act()
            steps_taken += 1

            if result in (ResultOfStep.STARVED, ResultOfStep.ENCOUNTERED_LION):  # might need tree
                print(f"steps taken: {steps_taken}, cause of death: {result}")
                break

            if steps_taken > 500:
                print(f"steps taken: {steps_taken}, survived over 500 steps")
                break