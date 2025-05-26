from enum import Enum


class GameResult(Enum):
    COMPLETE = 0,
    EATEN_BY_LION = 1,
    STARVED = 2,
    OOT = 3

class RecordModel:

    def __init__(self, player_id: str, result: GameResult, steps_taken: int, food_at_end: int, env_type: str, env_parameter_string: str, shot_lions: int = 0):
        self.player_id = player_id
        self.result = result
        self.steps_taken = steps_taken
        self.food_at_end = food_at_end
        self.env_type = env_type
        self.env_parameter_string = env_parameter_string
        self.shot_lions = shot_lions

    def __str__(self):
        return f"player: {self.player_id}, result: {self.result}, steps: {self.steps_taken}, food: {self.food_at_end}, env type: {self.env_parameter_string}"

    @staticmethod
    def make_parameter_string(tree_ration: float, lion_ratio: float):
        return f"T{tree_ration * 100}L{lion_ratio * 100}"

    @staticmethod
    def determine_result(scenario, oot = False):
        if (scenario.is_alive and scenario.steps_left > 0) or scenario.is_won():
            if oot:
                return GameResult.OOT.name
            return GameResult.COMPLETE.name
        elif not scenario.is_alive and scenario.steps_left > 0:
            return GameResult.EATEN_BY_LION.name
        else:
            return GameResult.STARVED.name

class ScenarioModel:

    def __init__(self, board, spawn, parameter_string, env_type):
        self.board = board
        self.spawn = spawn
        self.parameter_string = parameter_string
        self.env_type = env_type

    @staticmethod
    def make_parameter_string(tree_ration: float, lion_ratio: float):
        return f"T{tree_ration*100}L{lion_ratio*100}"

class GeneticIndividualModel:
    """action_set here is an ActionHolder, it gets serialized normally"""

    def __init__(self, ind_id, action_set, env_type, parent_id, other_parent_id = None):
        self.ind_id = ind_id
        self.action_set = action_set
        self.env_type = env_type
        self.parent_id = parent_id
        self.other_parent_id = other_parent_id