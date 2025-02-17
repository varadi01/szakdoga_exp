from enum import Enum

class TileState(Enum):
    LAND = 0
    TREE = 1
    LION = 2


class Step(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    STAY = 4

class ExtendedStep(Enum):
    STAY = 4 #dep


class ResultOfStep(Enum):
    NOTHING = 0
    FOUND_TREE = 1
    EATEN_BY_LION = 2
    STARVED = 3

    SHOT_LION = 4

class ExtendedResultOfStep(Enum):
    SHOT_LION = 4 #dep


class Environment:
    # neighbouring tiles' states
    def __init__(self, up: TileState, right: TileState, down: TileState, left: TileState):
        self.up = up
        self.right = right
        self.down = down
        self.left = left

    def __eq__(self, other):
        return (self.right == other.right
                and self.left == other.left
                and self.up == other.up
                and self.down == other.down)

    def get_as_list(self) -> list[int]:
        """returns a list of integers!!!"""
        return [
            self.up.value,
            self.right.value,
            self.down.value,
            self.left.value
        ]
    @staticmethod
    def get_from_list(l):
        return Environment(
            TileState(l[0]),
            TileState(l[1]),
            TileState(l[2]),
            TileState(l[3])
        )


class Position:
    def __init__(self, x_coordinate, y_coordinate):
        self.x = x_coordinate
        self.y = y_coordinate

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y