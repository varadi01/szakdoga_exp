from impl.rl import CustomEnv


def test_env_obs():
    env = CustomEnv()
    print(env.observation_space)
    print(env.observation_space.sample())
    env.reset()
    res = env.step(0)
    print(res)


test_env_obs()