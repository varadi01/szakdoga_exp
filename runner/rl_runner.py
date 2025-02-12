from solutions.rl import Agent, CustomEnvForSimpleGame, CustomEnvForExtendedGame, DummyVecEnv
from stable_baselines3 import DQN, A2C

#!!!
#at least 10k timesteps are needed
#it seems harsher environments train them quicker, and much better (way less getting eaten)
#training for too long also seems to deteriorate them -> it seems that after a certain amount of training, it decreases then plateaus
# tweaking lr does not help
#!!!

#extended did a funny: maximize reward -> get 1 for staying, stay all the time

def main():
    # rl_agent = Agent(name="DQN_4_long", do_save=True, alg=DQN)
    rl_agent = Agent(name="A2C_ex_3_few_lions", do_save=False, env_type="extended")

    rl_agent.learn(10000)
    print("training finished")

    # rl_agent.load_model("A2C_1")
    # print("loaded model")

    for i in range(10):
        rl_agent.test()

    # agent = Agent(A2C, "test", False)
    # agent.learn(1000)
    # agent.evaluate(10)
    # agent.test()


if __name__ == '__main__':
    main()