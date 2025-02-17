from solutions.rl import Agent, CustomEnvForSimpleGame, CustomEnvForExtendedGame, DummyVecEnv
from stable_baselines3 import DQN, A2C

#!!!
#at least 10k timesteps are needed
#it seems harsher environments train them quicker, and much better (way less getting eaten)
#training for too long also seems to deteriorate them -> it seems that after a certain amount of training, it decreases then plateaus
# tweaking lr does not help
#!!!

#extended did a funny: maximize reward -> get 1 for staying, stay all the time

#giving low reward makes it not want trees much, giving too high makes it take risks

#dqn likes easier envs it seems, needs to train longer

# 1 - T30L50
# 2 - T30L60
# 3 - T40L30
# 4 - T40L50
# 5 - T40L10

# short - 1k timesteps
# none - 10k timesteps
# semi-long - 100k timesteps
# long - 1M timesteps

#analyze losses

def main():
    # rl_agent = Agent(name="DQN_4_long", do_save=False, alg=DQN)
    #rl_agent = Agent(name="A2C_ex_6_no-punish_higher-rewards_none-for-step", do_save=False, env_type="extended")


    #rl_agent = Agent(name="A2C_simple_1_short", do_save=True, env_type="simple")
    #rl_agent = Agent(name="DQN_simple_3_long", do_save=True, env_type="simple", alg=DQN)
    # rl_agent = Agent(name="A2C_ex_5_semi-long", do_save=True, env_type="extended")
    # rl_agent = Agent(name="DQN_ex_5_semi-long", do_save=True, env_type="extended", alg=DQN)

    #rl_agent.learn(1000000)
    print("training finished")

    # rl_agent.load_model("A2C_1")
    # print("loaded model")

    for i in range(30):
        rl_agent.test()

    # agent = Agent(A2C, "test", False)
    # agent.learn(1000)
    # agent.evaluate(10)
    # agent.test()


if __name__ == '__main__':
    main()