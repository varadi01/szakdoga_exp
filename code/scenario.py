from random import randint
from enum import Enum
import numpy as np

#Keywords: Board, Environment, Position, Tile, Player, Step(Move), Death(Starvation, Lion)
#   Number of steps(food?), Tree(food source), Lion,  

#TODO detect unavoidable death
TREE_RATIO = 0.4
LION_RATIO = 0.3

INITIAL_NUMBER_OF_STEPS = 5
STEPS_GAINED_ON_FINDING_TREE = 2

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
    OK = 0
    ENCOUNTERED_LION = 1
    STARVED = 2


class Environment:
    # neighbouring tiles' states
    def __init__(self, up: TileState, right: TileState, down: TileState, left: TileState):
        self.up = up
        self.right = right
        self.down = down
        self.left = left



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

##TODO BAD!!
def translate_step(player_position: Position, step: Step) -> Position:
    match step:
        case Step.UP:
            player_position.y -= 1
            
        case Step.RIGHT:
            player_position.x += 1

        case Step.DOWN:
            player_position.y += 1

        case Step.LEFT:
            player_position.x =- 1
            
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
            board[current_position.x][current_position.y] = TileState.LAND #TODO smth's worng i can FEEL IT
            return board

class Game:
    # 10by10 list representing tiles
    def __init__(self, num_of_steps = INITIAL_NUMBER_OF_STEPS):
        self.board = generate_board()
        self.player_position = place_player(self.board)
        #detect unavoidable death, re-roll board? #TODO
        self.steps_left = num_of_steps
        self.is_alive = True
    
    #get states of surrounding tiles
    def get_environment(self) -> Environment:
        return Environment(
            self.get_tile_at_position(translate_step(self.player_position, Step.UP)),
            self.get_tile_at_position(translate_step(self.player_position, Step.RIGHT)),
            self.get_tile_at_position(translate_step(self.player_position, Step.DOWN)),
            self.get_tile_at_position(translate_step(self.player_position, Step.LEFT))
        )

    def get_tile_at_position(self, pos: Position) -> TileState:
        return self.board[pos.x][pos.y]

    def detect_unavoidable_death(self) -> bool:
        #TODO use

        # simple surround
        en = self.get_environment()
        if en.up == TileState.LION and en.right == TileState.LION and en.down == TileState.LION and en.left == TileState.LION:
            return True
        return False
        # waterfall method ##don't work, lions/trees can also wall in the waterfall itself

    def make_step(self, step: Step) -> ResultOfStep: ## meh
        new_position = translate_step(self.player_position, step)
        self.player_position = new_position
        self.steps_left -= 1

        #check where we stepped?
        if self.get_tile_at_position(new_position) == TileState.LAND:
            if self.steps_left > 0:
                return ResultOfStep.OK
            return ResultOfStep.STARVED
        
        if self.get_tile_at_position(new_position) == TileState.TREE:
            self.steps_left += STEPS_GAINED_ON_FINDING_TREE
            #remove tree, place elsewhere
            #self.board[new_position[0]][new_position[1]] = TileState.LAND
            self.board = tree_consumed(self.board, self.player_position)
            
            return ResultOfStep.OK
        
        if self.get_tile_at_position(new_position) == TileState.LION:
            self.is_alive = False
            return ResultOfStep.ENCOUNTERED_LION
        
