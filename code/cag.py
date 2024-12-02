from random import randint
from enum import Enum
#from numpy import

TREE_RATIO = 0.3
LION_RATIO = 0.2

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

def generate_board() -> list[list[TileState]]:
    #TODO
    pass

def translate_step(player_position, step: Step) -> tuple[int, int]:
    new_position = player_position
    
    match(step):
        case Step.UP:
            new_position = (new_position[0], new_position[1] + 1)
            
        case Step.RIGHT:
            new_position = (new_position[0] + 1, new_position[1])

        case Step.DOWN:
            new_position = (new_position[0], new_position[1] - 1)

        case Step.LEFT:
            new_position = (new_position[0] - 1, new_position[1])
            
    if new_position[0] == 10:
        new_position = (0, new_position[1])
    if new_position[1] == 10:
        new_position = (new_position[0], 0)
        
    return new_position

def place_player(board) -> tuple[int, int]:
    while True:
        x = randint(0, 9)
        y = randint(0, 9)
        if board[x][y] not in [TileState.LION, TileState.TREE]:
            return x, y

class Context:
    #neighbouring tiles' states
    def __init__(self, top: TileState, right: TileState, bottom: TileState, left: TileState):
        self.top = top
        self.right = right
        self.bottom = bottom
        self.left = left

class Game:
    # 10by10 list representing tiles
    def __init__(self):
        self.board = generate_board()
        self.player_position = place_player(self.board)
        self.steps_left = INITIAL_NUMBER_OF_STEPS
        self.is_alive = True
    
    #get states of surrounding tiles
    def get_context(self) -> Context:
        x = self.player_position[0]
        y = self.player_position[1]
        return Context(
            self.board[x][y+1], # top
            self.board[x+1][y], # right
            self.board[x][y-1], # bottom
            self.board[x-1][y] # left
        )
    
    #do we return anything with make_step or just terminate or what?
    ##died cus of lion, died of starvation etc?
    
    def make_step(self, step: Step) -> ResultOfStep: ## meh
        new_position = translate_step(self.player_position, step)
        self.player_position = new_position
        #check where we stepped?
        if self.board[new_position[0]][new_position[1]] == TileState.LAND:
            self.steps_left -= 1
            if self.steps_left > 0:
                return ResultOfStep.OK
            return ResultOfStep.STARVED
        
        if self.board[new_position[0]][new_position[1]] == TileState.TREE:
            self.steps_left += STEPS_GAINED_ON_FINDING_TREE
            return ResultOfStep.OK
        
        if self.board[new_position[0]][new_position[1]] == TileState.TREE:
            self.is_alive = False
            return ResultOfStep.ENCOUNTERED_LION
        
        