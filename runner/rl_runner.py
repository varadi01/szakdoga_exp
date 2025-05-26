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
# long - 100k timesteps


#less_p is -10 p- is -100

#analyze losses

#need to be rewarded for nothing, cus he won't stay ever otherwise
#ex is 1-50-500-0 or p -1000, zero means no for nothing

def main():

    #review
    # isolated -------------------------------------------------------------

    #doesent step on lions as often when no reward for lion but reward for else
    #performs best when high reward for lion?

    #T50S0L100
    #T50S0L10
    #T5S0L10
    #T5S0L0
    # name = "A2C_simple_T30L30_T50S1L100"
    # rl_simple = Agent(name=name+"_1", do_save=True, env_type="simple")
    # rl_simple.learn(10000)
    # rl_simple = Agent(name=name+"_2", do_save=True, env_type="simple")
    # rl_simple.learn(10000)
    # rl_simple = Agent(name=name+"_3", do_save=True, env_type="simple")
    # rl_simple.learn(10000)

    #length test
    # name = "A2C_T5S0L10_short"
    # rl_simple_length_test1 = Agent(name=name+"_1", do_save=True, env_type="simple")
    # rl_simple_length_test1.learn(1000)
    # rl_simple_length_test2 = Agent(name=name + "_2", do_save=True, env_type="simple")
    # rl_simple_length_test2.learn(1000)
    # rl_simple_length_test3 = Agent(name=name + "_3", do_save=True, env_type="simple")
    # rl_simple_length_test3.learn(1000)

    # name = "DQN_T5S0L10_long_5lr"
    # rl_simple_length_test1 = Agent(alg=DQN, name=name+"_1", do_save=True, env_type="simple", learning_rate=0.0005)
    # rl_simple_length_test1.learn(100000)
    # rl_simple_length_test2 = Agent(alg=DQN, name=name + "_2", do_save=True, env_type="simple", learning_rate=0.0005)
    # rl_simple_length_test2.learn(100000)
    # rl_simple_length_test3 = Agent(alg=DQN, name=name + "_3", do_save=True, env_type="simple", learning_rate=0.0005)
    # rl_simple_length_test3.learn(100000)

    # for i in range(10):
    #     rl_simple_length_test.test()

    # rl_dqn_simple = Agent(name="DQN_simple_T30L30_T5S1L10_longx5", do_save=True, env_type="simple", alg=DQN)
    # rl_dqn_simple.learn(500000)

    #strat test!!!!!!
    # name = "A2C_punish_T10S1L20"
    # rl_simple_strat = Agent(name=name+"_1", do_save=True, env_type="simple")
    # rl_simple_strat.learn(100000)
    # rl_simple_strat2 = Agent(name=name + "_2", do_save=True, env_type="simple")
    # rl_simple_strat2.learn(100000)
    # rl_simple_strat3 = Agent(name=name + "_3", do_save=True, env_type="simple")
    # rl_simple_strat3.learn(100000)

    #given strat try for best
    # for i in range(1,6):
    #     name = "A2C_T10S1L1000"
    #     rl = Agent(name=name+f"_{i}", do_save=True, env_type="simple")
    #     rl.learn(100000)



    # for i in range(1,11):
    #     name = "DQN_long_5xlr_T5S0L1"
    #     rl = Agent(name=name+f"_{i}", alg=DQN, do_save=True, env_type="simple", learning_rate=0.0005)
    #     rl.learn(100000)
    #     name = "DQN_medium_5xlr_T5S0L1"
    #     rl_med = Agent(name=name + f"_{i}", alg=DQN, do_save=True, env_type="simple", learning_rate=0.0005)
    #     rl_med.learn(10000)


    # for i in range(1,6):
    #     name = "A2C_ex_T1S0L0K2"
    #     rl_a = Agent(name=name+f"_{i}", do_save=True, env_type="extended")
    #     rl_a.learn(100000)
    #     name = "DQN_ex_long_T1S0L0K2"
    #     rl = Agent(name=name+f"_{i}", alg=DQN, do_save=True, env_type="extended", learning_rate=0.0005)
    #     rl.learn(100000)
    #     name = "DQN_ex_medium_T1S0L0K2"
    #     rl_med = Agent(name=name + f"_{i}", alg=DQN, do_save=True, env_type="extended", learning_rate=0.0005)
    #     rl_med.learn(10000)

    # end isolated -------------------------------------------------------------

    # rl_agent = Agent(name="A2C_simple_1_short", do_save=False, env_type="simple")
    #rl_agent = Agent(name="DQN_simple_3_long", do_save=True, env_type="simple", alg=DQN)
    # rl_agent = Agent(name="A2C_ex_1_less-p", do_save=True, env_type="extended")
    # rl_agent = Agent(name="DQN_ex_5_semi-long_zero_less-p", do_save=True, env_type="extended", alg=DQN)
    #
    # rl_agent.learn(100000)
    # print("training finished")
    #
    # # # rl_agent.load_model("A2C_1")
    # # # print("loaded model")
    # #
    # for i in range(3):
    #     rl_agent.test()

    # agent = Agent(A2C, "test", False)
    # agent.learn(1000)
    # agent.evaluate(10)
    # agent.test()

    pass


if __name__ == '__main__':
    main()