from impl.rl import CustomEnv
import stable_baselines3.common.env_checker as ch

def test_env_obs():
    env = CustomEnv()
    print(env.observation_space)
    print(env.observation_space.sample())
    env.reset()
    res = env.step(0)
    print(res)
    ch.check_env(env, True, True)


test_env_obs()