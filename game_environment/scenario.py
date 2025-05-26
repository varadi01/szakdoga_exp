import random
from random import randint
import numpy as np
from utils.scenario_utils import TileState, Step, ExtendedStep, ResultOfStep, ExtendedResultOfStep, Position, Environment

#Keywords: Board, Environment, Position, Tile, Player, Step(Move), Death(Starvation, Lion)
#   Number of steps(food?), Tree(food source), Lion,

#shooting a lion shouldnt cost food


#dep idea:
#   - have different trees, which give different food values
#   - punish backtracking

# dep IDEA:
#   - after choosing a step to take, give the model the upcoming environment, and see where it steps;
#       if it backtracks - make it choose a different step???????????????

#IDEA: actor can decide to wait?
TREE_RATIO = 0.3
LION_RATIO = 0.3

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
    def __init__(self, num_of_steps = INITIAL_NUMBER_OF_STEPS,
                 board = None, spawn: Position = None,
                 lion_ratio = LION_RATIO, tree_ratio = TREE_RATIO,
                 food_on_tree = STEPS_GAINED_ON_FINDING_TREE):
        self.TREE_RATIO = tree_ratio
        self.LION_RATIO = lion_ratio
        if board is None:
            self.board = self._generate_board()
        else:
            self.board = board
        if spawn is None:
            self.player_position = SimpleGame._place_player(self.board)
        else:
            self.player_position = spawn
        #re-generate board if death would be unavoidable
        while self._detect_unavoidable_death():
            self.board = self._generate_board()
            self.player_position = self._place_player(self.board)

        self.steps_left = num_of_steps
        self.food_on_tree = food_on_tree
        self.is_alive = True

    def get_environment(self) -> Environment:
        return Environment(
            self._get_tile_at_position(self._translate_step(self.player_position, Step.UP)),
            self._get_tile_at_position(self._translate_step(self.player_position, Step.RIGHT)),
            self._get_tile_at_position(self._translate_step(self.player_position, Step.DOWN)),
            self._get_tile_at_position(self._translate_step(self.player_position, Step.LEFT))
        )

    def make_step(self, step: Step) -> ResultOfStep:
        new_position = SimpleGame._translate_step(self.player_position, step)
        self.player_position = new_position
        self.steps_left -= 1

        #check where we stepped
        if self._get_tile_at_position(new_position) == TileState.LAND:
            if self.steps_left > 0:
                return ResultOfStep.NOTHING
            self.is_alive = False
            return ResultOfStep.STARVED

        if self._get_tile_at_position(new_position) == TileState.TREE:
            self.steps_left += self.food_on_tree
            #remove tree, place elsewhere
            self._tree_consumed()
            return ResultOfStep.FOUND_TREE

        if self._get_tile_at_position(new_position) == TileState.LION:
            self.is_alive = False
            return ResultOfStep.EATEN_BY_LION

    def _get_tile_at_position(self, pos: Position) -> TileState:
        return self.board[pos.x][pos.y]

    def _detect_unavoidable_death(self) -> bool:
        # simple surround
        en = self.get_environment()
        if en == Environment(TileState.LION,TileState.LION,TileState.LION,TileState.LION):
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

    def _generate_board(self) -> list[list[TileState]]:
        gen = np.random.choice([0, 1, 2],
                               (10, 10), True,
                               [1 - self.TREE_RATIO - self.LION_RATIO, self.TREE_RATIO, self.LION_RATIO])
        board = []
        for row in gen:
            r = []
            for cell in row:
                r.append(TileState(cell))  # probs?
            board.append(r)
        return board

    def is_won(self):
        return False

    @staticmethod
    def _translate_step(current_position: Position, step: Step) -> Position:
        translated_position = Position(current_position.x, current_position.y)
        match step:
            case Step.UP:
                translated_position.y -= 1

            case Step.RIGHT:
                translated_position.x += 1

            case Step.DOWN:
                translated_position.y += 1

            case Step.LEFT:
                translated_position.x -= 1

        if translated_position.x == 10:
            translated_position.x = 0
        if translated_position.y == 10:
            translated_position.y = 0
        if translated_position.x == -1:
            translated_position.x = 9
        if translated_position.y == -1:
            translated_position.y = 9

        return translated_position

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
##end

SimpleGame(tree_ratio=0.5, lion_ratio=0.4)


# extended scenario ideas:
# 1
# actor sees the results of actions and chooses from those? (might still be useful)


# 2!
# lions move randomly (maybe not every time (random chance / take some - rest pattern)
# new goal, other than survival
# if only 1 lion is around, the actor can step ON them to shoot and kill(remove) them

# choose
## (stay action, if the actor moves when in proximity of 2 or more lions, they die)
## #- kinda bloat-y, we'd get a lot of situations where stay is the best
## #- punish staying too long? lose food even when staying? dunno

## (pursuit, if the actor moves when in proximity of 2 or more lions, they get followed)
## #- might be hard to implement
## #- limits the freedom of movement of the actor, can get cornered (additional challenge?)
## #- might be interesting bc, we could aggro and 'lure' single lions to kill them
### #- pursuit thing might be better, if the lions wouldn't normally move?

# removing all lions means the game is won


## rewards difficult? training data difficult?
## more food from trees? as killing lions also consumes food


#### just because you can write an algorithm to solve the game, doesn't make it meaningless to make these!

class ExtendedGame(SimpleGame):

    def __init__(self, num_of_steps=INITIAL_NUMBER_OF_STEPS, board=None, spawn: Position = None,
                 lion_ratio=LION_RATIO, tree_ratio=TREE_RATIO,
                 food_on_tree=STEPS_GAINED_ON_FINDING_TREE):
        super().__init__(num_of_steps, board, spawn, lion_ratio, tree_ratio, food_on_tree)
        self.is_alive = True

    def make_step(self, step: Step) -> ResultOfStep:
        result = None
        if self._should_stay() and step != Step.STAY:
            self.is_alive = False
            return ResultOfStep.EATEN_BY_LION
        if step == Step.STAY:
            self.steps_left -= 0.25  # need to deduct some
            if self.steps_left > 0:
                result = ResultOfStep.NOTHING
            else:
                self.is_alive = False
                return ResultOfStep.STARVED
        new_position = SimpleGame._translate_step(self.player_position, step)
        tile_state = self._get_tile_at_position(new_position)
        old_player_position = self.player_position
        self.player_position = new_position
        if result is None:
            self.steps_left -= 1
            match tile_state:
                case TileState.LAND:
                    if self.steps_left > 0:
                        result = ResultOfStep.NOTHING
                    else:
                        self.is_alive = False
                        return ResultOfStep.STARVED

                case TileState.TREE:
                    self.steps_left += self.food_on_tree
                    self._tree_consumed()
                    result = ResultOfStep.FOUND_TREE

                case TileState.LION:
                    self._remove_lion(new_position)
                    self.steps_left += 1
                    result = ResultOfStep.SHOT_LION

        self._move_lions(old_player_position)
        return result

    def _move_lions(self, old_player_position: Position):
        # every time / random chance / take some - rest pattern?
        lion_positions = []

        for line in range(10):
            for col in range(10):
                pos = Position(line, col)
                if self._get_tile_at_position(pos) == TileState.LION:
                    lion_positions.append(pos)

        for lion_pos in lion_positions:
            possible_steps = [Step.UP, Step.LEFT, Step.DOWN, Step.RIGHT]
            new_pos = lion_pos
            while True:
                step = random.choice(possible_steps)
                new_proposed_pos = SimpleGame._translate_step(lion_pos, step)
                if (self._get_tile_at_position(new_proposed_pos) == TileState.LAND
                        and new_proposed_pos != self.player_position
                        # and new_proposed_pos != old_player_position
                    ):
                    new_pos = new_proposed_pos
                    break
                else:
                    possible_steps.remove(step)
                    if len(possible_steps) == 0:
                        break

            self.board[lion_pos.x][lion_pos.y] = TileState.LAND
            self.board[new_pos.x][new_pos.y] = TileState.LION

    def is_won(self) -> bool:
        """returns True if no lions remain on the board, False otherwise"""
        no_lions = True
        for line in self.board:
            for tile in line:
                if tile == TileState.LION:
                    no_lions = False
        return no_lions

    def _remove_lion(self, position: Position):
        self.board[position.x][position.y] = TileState.LAND

    def _valid_shot(self):
        #kinda redundant, as we cant reach here if it's not a valid shot, don't remove anyway
        env = self.get_environment()
        lion_count = 0
        for tile in env.get_as_list():
            if tile == TileState.LION.value:
                lion_count += 1
        return lion_count == 1

    def _should_stay(self):
        env = self.get_environment()
        lion_count = 0
        for tile in env.get_as_list():
            if tile == TileState.LION.value:
                lion_count += 1
        return lion_count > 1