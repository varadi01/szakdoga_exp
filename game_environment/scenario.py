from random import randint
import numpy as np
from utils.scenario_utils import TileState, Step, ResultOfStep, Position, Environment

#Keywords: Board, Environment, Position, Tile, Player, Step(Move), Death(Starvation, Lion)
#   Number of steps(food?), Tree(food source), Lion,



#todo idea:
#   - have different trees, which give different food values
#   - punish backtracking

# TODO IDEA:
#   - after choosing a step to take, give the model the upcoming environment, and see where it steps;
#       if it backtracks - make it choose a different step???????????????

#IDEA: actor can decide to wait?

TREE_RATIO = 0.4
LION_RATIO = 0.2

INITIAL_NUMBER_OF_STEPS = 5
STEPS_GAINED_ON_FINDING_TREE = 2

CONTEXT_WINDOW_LENGTH = 2


#deprecated
class PieceOfContext:

    def __init__(self, environment: Environment, step_taken: Step):
        self.environment = environment
        self.step_taken = step_taken

    def __eq__(self, other):
        return self.environment == other.environment and self.step_taken == other.step_taken

class ContextHolder:

    def __init__(self, context: list[PieceOfContext] = None, context_window_length: int = CONTEXT_WINDOW_LENGTH):
        if context is None:
            context = []
        self.context = context
        self.window_size = context_window_length

    def get_context(self):
        return self.context

    def get_nth_context_piece(self, n: int):
        try:
            return self.context[n]
        except IndexError:
            return None

    def update_context(self, environment: Environment, step: Step):
        self.context.insert(0, PieceOfContext(environment, step))  # latest first
        if len(self.context) > self.window_size:
            self.context.pop()

    def update_context_with_piece(self, context_piece: PieceOfContext):
        self.context.insert(0, context_piece)  # latest first
        if len(self.context) > self.window_size:
            self.context.pop()

    @staticmethod
    def contexts_eq( a_context: list[PieceOfContext], other_context: list[PieceOfContext]) -> bool:
        if len(a_context) != len(other_context):
            return False
        for i in range(len(a_context)):
            if a_context[i] != other_context[i]:
                return False
        return True
#end deprecated

class SimpleGame:
    # 10by10 list representing tiles
    def __init__(self, num_of_steps = INITIAL_NUMBER_OF_STEPS):
        self.board = self._generate_board()
        self.player_position = self._place_player(self.board)
        #re-generate board if death would be unavoidable
        while self._detect_unavoidable_death():
            self.board = self._generate_board()
            self.player_position = self._place_player(self.board)

        self.steps_left = num_of_steps
        self.is_alive = True

    def get_environment(self) -> Environment:
        return Environment(
            self._get_tile_at_position(self._translate_step(self.player_position, Step.UP)),
            self._get_tile_at_position(self._translate_step(self.player_position, Step.RIGHT)),
            self._get_tile_at_position(self._translate_step(self.player_position, Step.DOWN)),
            self._get_tile_at_position(self._translate_step(self.player_position, Step.LEFT))
        )

    def make_step(self, step: Step) -> ResultOfStep:
        new_position = self._translate_step(self.player_position, step)
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
            self._tree_consumed()
            return ResultOfStep.TREE

        if self._get_tile_at_position(new_position) == TileState.LION:
            self.is_alive = False
            return ResultOfStep.ENCOUNTERED_LION

    def _get_tile_at_position(self, pos: Position) -> TileState:
        return self.board[pos.x][pos.y]

    def _detect_unavoidable_death(self) -> bool:
        # simple surround
        en = self.get_environment()
        if en.up == TileState.LION and en.right == TileState.LION and en.down == TileState.LION and en.left == TileState.LION:
            return True
        return False

    def _tree_consumed(self):
        while True:
            x = randint(0, 9)
            y = randint(0, 9)
            if self.board[x][y] not in [TileState.LION, TileState.TREE] and Position(x, y) != self.player_position:
                self.board[x][y] = TileState.TREE
                self.board[self.player_position.x][self.player_position.y] = TileState.LAND
                break


    @staticmethod
    def _generate_board() -> list[list[TileState]]:
        gen = np.random.choice([0, 1, 2],
                               (10, 10), True,
                               [1 - TREE_RATIO - LION_RATIO, TREE_RATIO, LION_RATIO])
        board = []
        for row in gen:
            r = []
            for cell in row:
                r.append(TileState(cell))  # probs?
            board.append(r)
        return board

    @staticmethod
    def _translate_step(current_position: Position, step: Step) -> Position:
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

    @staticmethod
    def _place_player(board) -> Position:
        while True:
            x = randint(0, 9)
            y = randint(0, 9)
            if board[x][y] not in [TileState.LION, TileState.TREE]:
                return Position(x, y)

#deprecated
class ContextBasedGame(SimpleGame):

    def __init__(self, num_of_steps=INITIAL_NUMBER_OF_STEPS, context_window_length=CONTEXT_WINDOW_LENGTH):
        super().__init__(num_of_steps)
        self.context: ContextHolder = ContextHolder(context_window_length=context_window_length)

    def get_context(self):
        return self.context.get_context()

    def make_step(self, step: Step) -> ResultOfStep:
        prev = self.get_environment()
        res = super().make_step(step)
        #update context
        self.context.update_context(prev, step)
        return res