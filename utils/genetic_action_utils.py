from game_environment.scenario import Environment, Step

class Action:

    def __init__(self, env: Environment, step: Step):
        self.env = env
        self.step = step

    def env_eq(self, other):
        return self.env == other.env

    def __eq__(self, other):
        return self.env == other.env and self.step == other.step


class ActionHolder:

    def __init__(self, actions: list[Action] = None):
        if actions is None:
            actions = []
        self.actions = actions

    def add_action(self, action: Action):
        if self.is_env_known(action.env):
            return
        self.actions.append(action)

    def remove_action(self, action):
        try:
            self.actions.remove(action)
        except ValueError:
            raise Exception("tried to delete non-existent action")

    def is_env_known(self, env: Environment) -> bool:
        known = False
        for act in self.actions:
            if act.env == env:
                known = True
        return known

    def get_action_for_env(self, env: Environment) -> Action:
        action = None
        for act in self.actions:
            if act.env == env:
                action = act
        if action is not None:
            return action
        raise Exception("action not known")
