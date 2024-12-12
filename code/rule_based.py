from scenario import Game, Step, Environment, TileState, ResultOfStep
from random import choice

#TODO apparently we're stepping on lions

class RuleBasedPlayer:
    #gets instantiated and called by runner
    def __init__(self, game: Game):
        self.game = game

    def act(self) -> tuple[ResultOfStep, Step]:
        #get env
        environment = self.game.get_environment()
        print(f"up:{environment.up}, right:{environment.right}, down:{environment.down}, left:{environment.left}")
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
            return self.game.make_step(c), c
        if len(land) > 0: #should always be true?
            cl = choice(land)
            print(f"step {cl}")
            return self.game.make_step(cl), cl
        return choice((Step.RIGHT, Step.LEFT, Step.UP, Step.DOWN))