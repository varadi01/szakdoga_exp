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

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from utils.scenario_utils import Step, Environment, ResultOfStep
from game_environment.scenario import SimpleGame


#TODO make DQN work aswell

REWARD_FOR_TREE = 30
REWARD_FOR_LAND = 1
REWARD_FOR_LION = -100
REWARD_FOR_STARVED = -10 #prolly dont need this?

class CustomEnv(gym.Env):
    """Custom environment of the game for agents to operate on"""

    def __init__(self):
        self.action_space = gym.spaces.Discrete(4)
        # might also be a tuple of discrete spaces, maybe simpler this way; Box might also work
        self.observation_space = gym.spaces.MultiDiscrete([4, 4, 4, 4])
        self.scenario = None

    def _get_obs(self):
        env = self.scenario.get_environment()
        #TODO test
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
            case ResultOfStep.LAND:
                reward = REWARD_FOR_LAND
            case ResultOfStep.TREE:
                reward = REWARD_FOR_TREE
            case ResultOfStep.STARVED:
                reward = REWARD_FOR_STARVED
                terminated = True
            case ResultOfStep.ENCOUNTERED_LION:
                reward = REWARD_FOR_LION
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
        self.scenario = SimpleGame()
        info = self._get_info()
        return self._get_obs(), info

    def close(self):
        """clean up used resources, close renderer"""
        return NotImplementedError("we don't do that here")

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        """render a view of episodes"""
        return NotImplementedError("we don't do that here")


class Agent:

    #give it a name, if it exists load it, if not create
    #should make model a field too
    def __init__(self, alg, name: str, do_save: bool = False):
        self.alg = alg
        self.name = name
        self.do_save = do_save
        self.model = None

    def learn(self, timesteps: int):
        print("started learn")
        env = CustomEnv()
        env = DummyVecEnv([lambda: env]) #only vectorize for A2C
        m = self.alg('MlpPolicy', env, verbose = 1) #check policy
        m.learn(total_timesteps = timesteps)
        self.model = m
        print("ended learn")

        if self.do_save:
            print("saving model")
            self._save_model(m)

    def evaluate(self, episodes: int):
        #dont fully understand this one
        env = CustomEnv()
        env = DummyVecEnv([lambda: env]) #only vectorize for A2C

        evaluate_policy(self.model, env, n_eval_episodes=episodes, render=False)

    def test(self):
        env = CustomEnv()
        obs, _ = env.reset()
        print("testing agent")
        steps_taken = 0 #TODO temp
        while True:
            action, _states = self.model.predict(obs)
            obs, rewards, done, _, info = env.step(action)
            steps_taken += 1
            print(f"action taken: {Step(action)}, reward gotten: {rewards}, info: {info}")
            if done:
                print(f"done, with {steps_taken} steps taken")
                break

    def _save_model(self, model):
        path = os.path.join('rl', 'models', self.name)
        model.save(path)

    #TODO use
    def _load_model(self, model, env):
        path = os.path.join('rl', 'models', self.name)
        if not os.path.isfile(path):
            FileNotFoundError("no model with such a name exists")
        model.load(path, env=env) #might not be path