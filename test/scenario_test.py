from impl.scenario import SimpleGame, Position, Step


def test_translate_step():
    pos = Position(1,1)

    new_pos = translate_step(pos, Step.RIGHT)
    if new_pos == Position(2,1):
        print("1 - passed")
    else:
        print("1 - fail")

    new_pos = translate_step(pos, Step.DOWN)
    if new_pos == Position(1, 2):
        print("2 - passed")
    else:
        print("2 - fail")

    new_pos = translate_step(pos, Step.LEFT)
    if new_pos == Position(0, 1):
        print("3 - passed")
    else:
        print("3 - fail")

    new_pos = translate_step(pos, Step.UP)
    if new_pos == Position(1, 0):
        print("4 - passed")
    else:
        print("4 - fail")

    pos2 = Position(9,9)

    new_pos = translate_step(pos2, Step.RIGHT)
    if new_pos == Position(0, 9):
        print("1 - passed")
    else:
        print("1 - fail")

    new_pos = translate_step(pos2, Step.DOWN)
    if new_pos == Position(9, 0):
        print("2 - passed")
    else:
        print("2 - fail")

    new_pos = translate_step(pos2, Step.LEFT)
    if new_pos == Position(8, 9):
        print("3 - passed")
    else:
        print("3 - fail")

    new_pos = translate_step(pos2, Step.UP)
    if new_pos == Position(9, 8):
        print("4 - passed")
    else:
        print("4 - fail")


test_translate_step()