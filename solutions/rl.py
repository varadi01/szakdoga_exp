# self-made environment - discrete, agent should operate with it
# episodes are a run of the game
# rewards: land - ,tree - ,lion -
# action-space, observation-space both discrete(4)

##https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html

from typing import Tuple, Optional, Union, List
import os

import gym

from gym.core import ActType, ObsType, RenderFrame

import numpy as np

from stable_baselines3 import DQN, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from utils.scenario_utils import Step, ExtendedStep, ResultOfStep, ExtendedResultOfStep
from game_environment.scenario import SimpleGame, ExtendedGame


REWARD_FOR_FINDING_TREE = 1
REWARD_FOR_TAKING_STEP = 0
REWARD_FOR_GETTING_EATEN = 0
REWARD_FOR_STARVING = 0 #prolly dont need this?

class CustomEnvForSimpleGame(gym.Env):
    """Custom environment of the game for agents to operate on"""

    def __init__(self, scenario = None, specific_scenario = False):
        self.action_space = gym.spaces.Discrete(4)
        # might also be a tuple of discrete spaces, maybe simpler this way; Box might also work
        self.observation_space = gym.spaces.MultiDiscrete([4, 4, 4, 4])
        self.scenario = scenario
        self.specific_scenario = specific_scenario

    def _get_obs(self):
        env = self.scenario.get_environment()
        return np.array(env.get_as_list())

    def _take_action(self, action) -> ResultOfStep:
        step = Step(action)
        return self.scenario.make_step(step)

    def _get_info(self):
        """UNIMPLEMENTED"""
        return {} #not used for anything currently

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        """takes an action in the episode"""
        result = self._take_action(action)
        reward = 0
        terminated = False
        match result:
            case ResultOfStep.NOTHING:
                reward = REWARD_FOR_TAKING_STEP
            case ResultOfStep.FOUND_TREE:
                reward = REWARD_FOR_FINDING_TREE
            case ResultOfStep.STARVED:
                reward = REWARD_FOR_STARVING
                terminated = True
            case ResultOfStep.EATEN_BY_LION:
                reward = REWARD_FOR_GETTING_EATEN
                terminated = True

        new_obs = self._get_obs()
        info = self._get_info()
        return new_obs, reward, terminated, False, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[ObsType, dict]:
        """Instantiates a new episode"""
        super().reset()
        if not self.specific_scenario:
            self.scenario = SimpleGame()
        info = self._get_info()
        return self._get_obs(), info

    def close(self):
        """clean up used resources, close renderer"""
        return NotImplementedError("we don't do that here")

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        """render a view of episodes"""
        return NotImplementedError("we don't do that here")

REWARD_FOR_SHOOTING_LION = 2

class CustomEnvForExtendedGame(gym.Env):
    """Custom environment of the game for agents to operate on"""

    def __init__(self, scenario = None, specific_scenario = False):
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.MultiDiscrete([4, 4, 4, 4])
        self.scenario: ExtendedGame = scenario
        self.specific_scenario = specific_scenario

    def _get_obs(self):
        env = self.scenario.get_environment()
        return np.array(env.get_as_list())

    def _take_action(self, action) -> ResultOfStep:
        step = Step(action)
        return self.scenario.make_step(step)

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        """takes an action in the episode"""
        result = self._take_action(action)
        reward = 0
        terminated = False
        match result:
            case ResultOfStep.NOTHING:
                reward = REWARD_FOR_TAKING_STEP #- (REWARD_FOR_TAKING_STEP + 1)
            case ResultOfStep.FOUND_TREE:
                reward = REWARD_FOR_FINDING_TREE
            case ResultOfStep.STARVED:
                reward = REWARD_FOR_STARVING
                terminated = True
            case ResultOfStep.EATEN_BY_LION:
                reward = REWARD_FOR_GETTING_EATEN
                terminated = True
            case ResultOfStep.SHOT_LION:
                reward = REWARD_FOR_SHOOTING_LION

        if self.scenario.is_won():
            terminated = True

        new_obs = self._get_obs()
        info = self._get_info()
        return new_obs, reward, terminated, False, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[ObsType, dict]:
        """Instantiates a new episode"""
        super().reset()
        if not self.specific_scenario:
            self.scenario = ExtendedGame()
        info = self._get_info()
        return self._get_obs(), info

    def _get_info(self):
        """UNIMPLEMENTED"""
        return {} #not used for anything currently

    def close(self):
        """clean up used resources, close renderer"""
        return NotImplementedError("we don't do that here")

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        """render a view of episodes"""
        return NotImplementedError("we don't do that here")

class Agent:

    def __init__(self, alg = A2C, name: str = "test", do_save: bool = False, env_type: str = "simple", learning_rate: float = None):
        self.alg = alg
        self.name = name
        self.do_save = do_save
        self.model = None
        self.env_type = env_type
        self.learning_rate = learning_rate

    def learn(self, timesteps: int):
        print("started learn")
        env = CustomEnvForSimpleGame()
        if self.env_type == "extended":
            env = CustomEnvForExtendedGame()
        if self.alg == A2C:
            env = DummyVecEnv([lambda: env])
        if self.learning_rate is not None:
            m = self.alg('MlpPolicy', env, verbose = 1, learning_rate=self.learning_rate)
        else:
            m = self.alg('MlpPolicy', env, verbose=1)
        m.learn(total_timesteps = timesteps)
        self.model = m
        print("ended learn")

        if self.do_save:
            print("saving model")
            self.save_model(m)

    def evaluate(self, episodes: int):
        env = CustomEnvForSimpleGame()
        if self.env_type == "extended":
            env = CustomEnvForExtendedGame()
        if self.alg == A2C:
            env = DummyVecEnv([lambda: env])
        evaluate_policy(self.model, env, n_eval_episodes=episodes, render=False)

    def test(self):
        env = CustomEnvForSimpleGame()
        if self.env_type == "extended":
            env = CustomEnvForExtendedGame()
        obs, _ = env.reset()
        steps_made = 0
        cumulated_reward = 0
        while True:
            action, _states = self.model.predict(obs)
            # print(f"obs: {obs}")
            obs, rewards, done, _, info = env.step(action)
            # print(f"action taken: {Step(action)}, reward gotten: {rewards},food: {env.scenario.steps_left}, info: {info}")
            cumulated_reward += rewards
            steps_made += 1
            if done:
                print(f"done, with {steps_made} steps taken, cumulative reward {cumulated_reward}")
                break

    def save_model(self, model):
        path = os.path.join(os.getcwd(), "..", 'rl', 'models_4_ex', self.name) #changed!
        print(path)
        model.save(path)

    def load_model(self, model_name: str, env = None):
        path = os.path.join(os.getcwd(), '..', 'rl', 'models_4_ex', model_name) #changed!
        print(path)
        # if not os.path.isfile(path):
        #     FileNotFoundError("no model with such a name exists")
        self.model = self.alg.load(path, env=env) #might not be path