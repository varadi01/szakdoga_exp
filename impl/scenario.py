from random import randint
from enum import Enum
import numpy as np

#Keywords: Board, Environment, Position, Tile, Player, Step(Move), Death(Starvation, Lion)
#   Number of steps(food?), Tree(food source), Lion,  

TREE_RATIO = 0.5
LION_RATIO = 0.1

INITIAL_NUMBER_OF_STEPS = 5
STEPS_GAINED_ON_FINDING_TREE = 2

CONTEXT_WINDOW_LENGTH = 2

class TileState(Enum):
    LAND = 0
    TREE = 1
    LION = 2
    
class Step(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    
class ResultOfStep(Enum):
    LAND = 0
    TREE = 1
    ENCOUNTERED_LION = 2
    STARVED = 3

class Environment:
    # neighbouring tiles' states
    def __init__(self, up: TileState, right: TileState, down: TileState, left: TileState):
        self.up = up
        self.right = right
        self.down = down
        self.left = left

    def __eq__(self, other):
        if self.right == other.right and  self.left == other.left and  self.up == other.up and  self.down == other.down:
            return True
        return False

class Position:
    def __init__(self, x_coordinate, y_coordinate):
        self.x = x_coordinate
        self.y = y_coordinate

    def __eq__(self, other):
        if self.x == other.x and self.y == other.y:
            return True
        return False

def generate_board() -> list[list[TileState]]:
    gen = np.random.choice([0, 1, 2],
                           (10, 10), True,
                           [1-TREE_RATIO-LION_RATIO, TREE_RATIO, LION_RATIO])
    board = []
    for row in gen:
        r = []
        for cell in row:
            r.append(TileState(cell)) #probs?
        board.append(r)
    return board

def translate_step(current_position: Position, step: Step) -> Position:
    player_position = Position(current_position.x, current_position.y)
    match step:
        case Step.UP:
            player_position.y -= 1
            
        case Step.RIGHT:
            player_position.x += 1

        case Step.DOWN:
            player_position.y += 1

        case Step.LEFT:
            player_position.x -= 1

    if player_position.x == 10:
        player_position.x = 0
    if player_position.y == 10:
        player_position.y = 0
    if player_position.x == -1:
        player_position.x = 9
    if player_position.y == -1:
        player_position.y = 9

    return player_position

def place_player(board) -> Position:
    while True:
        x = randint(0, 9)
        y = randint(0, 9)
        if board[x][y] not in [TileState.LION, TileState.TREE]:
            return Position(x, y)
        
def tree_consumed(board, current_position: Position):
    while True:
        x = randint(0, 9)
        y = randint(0, 9)
        if board[x][y] not in [TileState.LION, TileState.TREE] and Position(x, y) != current_position:
            board[x][y] = TileState.TREE
            board[current_position.x][current_position.y] = TileState.LAND
            return board

class SimpleGame:
    # 10by10 list representing tiles
    def __init__(self, num_of_steps = INITIAL_NUMBER_OF_STEPS):
        self.board = generate_board()
        self.player_position = place_player(self.board)
        #re-generate board if death would be unavoidable
        while self._detect_unavoidable_death():
            self.board = generate_board()
            self.player_position = place_player(self.board)

        self.steps_left = num_of_steps
        self.is_alive = True

    def get_environment(self) -> Environment:
        return Environment(
            self._get_tile_at_position(translate_step(self.player_position, Step.UP)),
            self._get_tile_at_position(translate_step(self.player_position, Step.RIGHT)),
            self._get_tile_at_position(translate_step(self.player_position, Step.DOWN)),
            self._get_tile_at_position(translate_step(self.player_position, Step.LEFT))
        )

    def _get_tile_at_position(self, pos: Position) -> TileState:
        return self.board[pos.x][pos.y]

    def _detect_unavoidable_death(self) -> bool:
        # simple surround
        en = self.get_environment()
        if en.up == TileState.LION and en.right == TileState.LION and en.down == TileState.LION and en.left == TileState.LION:
            return True
        return False

    def make_step(self, step: Step) -> ResultOfStep:
        new_position = translate_step(self.player_position, step)
        self.player_position = new_position
        self.steps_left -= 1

        #check where we stepped
        if self._get_tile_at_position(new_position) == TileState.LAND:
            if self.steps_left > 0:
                return ResultOfStep.LAND
            self.is_alive = False
            return ResultOfStep.STARVED

        if self._get_tile_at_position(new_position) == TileState.TREE:
            self.steps_left += STEPS_GAINED_ON_FINDING_TREE
            #remove tree, place elsewhere
            self.board = tree_consumed(self.board, self.player_position)
            return ResultOfStep.TREE

        if self._get_tile_at_position(new_position) == TileState.LION:
            self.is_alive = False
            return ResultOfStep.ENCOUNTERED_LION
        

class ContextBasedGame(SimpleGame):

    def __init__(self, num_of_steps=INITIAL_NUMBER_OF_STEPS, context_window_length=CONTEXT_WINDOW_LENGTH):
        super().__init__(num_of_steps)
        self.context: list[tuple[Environment, Step]] = [] #TODO probs not okay, eq
        self.window_size = context_window_length

    def get_context(self):
        return self.context

    def make_step(self, step: Step) -> ResultOfStep:
        prev = self.get_environment()
        res = super().make_step(step)
        #update context
        self.context.insert(0, (prev, step)) #latest first
        if len(self.context) > self.window_size:
            self.context.pop()
        return res