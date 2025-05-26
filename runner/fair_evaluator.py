#gets set of scenarios, and evaluates solutions on each
import copy

import numpy as np
from torch.backends.cudnn import deterministic

from utils.db_context import get_instance
from stable_baselines3 import DQN, A2C
from utils.scenario_utils import Position, TileState, Step, Environment
from game_environment.scenario import SimpleGame, ExtendedGame
from utils.db_entities import ScenarioModel, GeneticIndividualModel, RecordModel, GameResult
from solutions.genetic import Individual, GeneticNaive, Genetic, ExtendedIndividual
from solutions.rl import CustomEnvForSimpleGame, CustomEnvForExtendedGame, Agent
from solutions.deepl import load_model, Deepl, ExtendedDeepl
from solutions.rule_based import RuleBasedPlayer, ExtendedRuleBasePlayer
from utils.genetic_action_utils import ActionHolder


def get_scenarios_from_collection(collection_name: str, env_type: str, db: str = "scenarios_ex_2"):
    db = get_instance(collection_name, 's', db) #changed!
    scs: list[ScenarioModel] = db.get_all()
    games = []
    if env_type == "simple":
        for s in scs:
            games.append(SimpleGame(
                board= construct_board(s.board),
                spawn= Position(s.spawn[0], s.spawn[1])
            ))
        print(f"found {len(games)} scenarios")
    elif env_type == "extended":
        for s in scs:
            games.append(ExtendedGame(
                board= construct_board(s.board),
                spawn= Position(s.spawn[0], s.spawn[1])
            ))
        print(f"found {len(games)} scenarios")
    return games

def get_individuals_from_collection(collection_name: str, env_type: str, db: str):
    db = get_instance(collection_name, 'g', db) #changed!
    ind_models = db.get_all()
    individuals = []
    if env_type == "simple":
        for model in ind_models:
            individuals.append(Individual(
                model.ind_id.split(',')[0],
                model.ind_id.split(',')[1],
                known_actions=model.action_set.actions,
                parent_id=model.parent_id,
                other_parent_id=model.other_parent_id
            ))

    elif env_type == "extended":
        for model in ind_models:
            individuals.append(ExtendedIndividual(
                model.ind_id.split(',')[0],
                model.ind_id.split(',')[1],
                known_actions=model.action_set.actions,
                parent_id=model.parent_id,
                other_parent_id=model.other_parent_id
            ))

    # for individual in individuals:
    #     for a in individual.known_actions.actions:
    #         if len(individual.known_actions.actions) > 80:
    #             print(f"more than 80 known actions: {len(individual.known_actions.actions)}")
            # for action in individual.known_actions.actions:
            #     if action.env == Environment(TileState.LION,TileState.LION,TileState.LION,TileState.LION):
            #         print()
            # if a.env.get_as_list()[a.step.value] == 2:
            #     print(f"bad step {a.env.get_as_list()} - {a.step.value} at {individual.id}")
            #     individuals.remove(individual)
    ft_individuals = [i for i in individuals if i.is_fully_trained()] #for prog
    return ft_individuals

def generate_scenarios(tree_ratio, lion_ratio, env_type, collection_name, db_name = "scenarios", num_of_scenarios = 10):
    if env_type == "simple":
        scenarios = []
        for _ in range(num_of_scenarios):
            scenarios.append(SimpleGame(lion_ratio=lion_ratio, tree_ratio=tree_ratio))
        models = []
        for s in scenarios:
            models.append(ScenarioModel(
                board= deconstruct_board(s.board),
                spawn= [s.player_position.x, s.player_position.y],
                parameter_string= ScenarioModel.make_parameter_string(tree_ratio, lion_ratio),
                env_type= env_type
            ))
        db = get_instance(collection_name, 's', db_name)
        db.insert_many(models)
    elif env_type == "extended":
        scenarios = []
        for _ in range(num_of_scenarios):
            scenarios.append(ExtendedGame(lion_ratio=lion_ratio, tree_ratio=tree_ratio))
        models = []
        for s in scenarios:
            models.append(ScenarioModel(
                board=deconstruct_board(s.board),
                spawn=[s.player_position.x, s.player_position.y],
                parameter_string=ScenarioModel.make_parameter_string(tree_ratio, lion_ratio),
                env_type=env_type
            ))
        db = get_instance(collection_name, 's', db_name)
        db.insert_many(models)


def construct_board(board):
    converted_board = []
    #probably? mirrored is fine too
    for line in board:
        c_line = []
        for cell in line:
            c_line.append(TileState(cell))
        converted_board.append(c_line)
    return converted_board

def deconstruct_board(board: list[list[TileState]]):
    converted_board = []
    for line in board:
        c_line = []
        for cell in line:
            c_line.append(cell.value)
        converted_board.append(c_line)
    return converted_board


def evaluate(scenario_collection: str, env_type: str,
            model_type: str, target_collection: str = None,
            model_path: str = None,
            gen_collection: str = None,
            is_dqn = False,
            eval_period = 1):
    scenarios = get_scenarios_from_collection(scenario_collection, env_type)
    if target_collection is None:
        target_collection = f"{env_type}-{model_type};{model_path}{gen_collection};{scenario_collection}"

    for _ in range(eval_period):
        match model_type:
            case "deepl":
                eval_deepl(scenarios, model_path, target_collection)
            case "rl":
                eval_rl(scenarios, model_path, target_collection, is_dqn)
            case "gen":
                eval_gen(scenarios, gen_collection, target_collection)
            case "deepl_ex":
                eval_deepl_extended(scenarios, model_path, target_collection)
            case "rl_ex":
                eval_rl_extended(scenarios, model_path, target_collection, is_dqn)
            case "gen_ex":
                eval_gen_extended(scenarios, gen_collection, target_collection)
            case "rule_based":
                eval_rule_based(scenarios, target_collection)
            case "rule_based_ex":
                eval_rule_based_extended(scenarios, target_collection)


def eval_deepl(scenarios, model_path, target_collection):
    model = load_model(model_path)
    records = []
    for scenario_o in scenarios:
        steps = 0
        scenario = copy.deepcopy(scenario_o)
        while scenario.is_alive:
            env = scenario.get_environment()
            prediction = model.predict(np.array(env.get_as_list())[None, ...], verbose=0)
            step_int = np.argmax(prediction)
            scenario.make_step(Step(step_int))
            steps += 1
            if steps >= 500:
                break
        records.append(RecordModel(
            "deepl",
            RecordModel.determine_result(scenario),
            steps,
            scenario.steps_left,
            "simple",
            ScenarioModel.make_parameter_string(scenario.TREE_RATIO, scenario.LION_RATIO)
        ))
    db = get_instance(target_collection, 'r', "records_deepl_3_batch") #changed!
    db.insert_many(records)

def eval_rl(scenarios, model_path, target_collection, is_dqn = False):
    if is_dqn:
        agent = Agent(name="eval", alg=DQN)
    else:
        agent = Agent(name="eval")
    records = []
    for scenario_o in scenarios:
        scenario = copy.deepcopy(scenario_o) #maybe dont need
        env = CustomEnvForSimpleGame(scenario, True)
        agent.load_model(model_path, env)
        obs, _ = env.reset()
        steps_taken = 0
        while True: #maybe is_alive, check
            action, _states = agent.model.predict(obs, deterministic=True) #changed!
            obs, rewards, done, _, info = env.step(action)
            steps_taken += 1
            if steps_taken >= 500:
                done = True
            if done:
                records.append(RecordModel(
                    "rl",
                    RecordModel.determine_result(env.scenario),
                    steps_taken,
                    scenario.steps_left,
                    "simple",
                    ScenarioModel.make_parameter_string(env.scenario.TREE_RATIO, env.scenario.LION_RATIO)
                ))
                break
    db = get_instance(target_collection, 'r', "records_rl_batch") #changed!
    db.insert_many(records)

def eval_gen(scenarios: list[SimpleGame], gen_collection, target_collection):
    individuals: list[Individual] = get_individuals_from_collection(gen_collection, "simple", "col_name") # changed!
    records = []
    for scenario in [s for s in scenarios]:
        for individual in [_ for _ in individuals]:
            scenario_copy: SimpleGame = copy.deepcopy(scenario)
            individual.set_specific_scenario(scenario_copy)
            while individual.scenario.is_alive:
                individual.act()
                if individual.steps_made >= 500:
                    break

            records.append(RecordModel(
                "genetic",
                RecordModel.determine_result(individual.scenario),
                individual.steps_made,
                individual.scenario.steps_left,
                "simple",
                ScenarioModel.make_parameter_string(individual.scenario.TREE_RATIO, individual.scenario.LION_RATIO)
            ))
    db = get_instance(target_collection, 'r', "records_gen_4_fix") #changed!
    db.insert_many(records)


def eval_deepl_extended(scenarios, model_path, target_collection):
    model = load_model(model_path)
    records = []
    for scenario_o in scenarios:
        steps = 0
        scenario = copy.deepcopy(scenario_o)
        oot = False
        while scenario.is_alive:
            env = scenario.get_environment()
            prediction = model.predict(np.array(env.get_as_list())[None, ...], verbose=0)
            step_int = np.argmax(prediction)
            scenario.make_step(Step(step_int))
            if Step(step_int) != Step.STAY:
                steps += 1
            if scenario.is_won():
                break
            if steps > 1000:
                oot = True
                break
        no_of_lions = 0
        for line in scenario.board:
            for tile in line:
                if tile == TileState.LION:
                    no_of_lions += 1
        shot_lions = scenario.LION_RATIO * 100 - no_of_lions
        records.append(RecordModel(
            "deepl_ex",
            RecordModel.determine_result(scenario, oot),
            steps,
            scenario.steps_left,
            "extended",
            ScenarioModel.make_parameter_string(scenario.TREE_RATIO, scenario.LION_RATIO),
            shot_lions
        ))
    db = get_instance(target_collection, 'r', "records_deepl_ex_final")
    db.insert_many(records)

def eval_rl_extended(scenarios, model_path, target_collection, is_dqn):
    if is_dqn:
        agent = Agent(name="eval", alg=DQN)
    else:
        agent = Agent(name="eval")
    records = []
    for scenario_o in scenarios:
        scenario = copy.deepcopy(scenario_o) #maybe dont need
        env = CustomEnvForExtendedGame(scenario, True)
        agent.load_model(model_path, env)
        obs, _ = env.reset()
        steps_taken = 0
        oot = False
        while True: #maybe is_alive, check
            action, _states = agent.model.predict(obs, deterministic=True)
            obs, rewards, done, _, info = env.step(action)
            if Step(action) != Step.STAY:
                steps_taken += 1
            if steps_taken > 1000:
                done = True
                oot = True
            if done:
                no_of_lions = 0
                for line in env.scenario.board:
                    for tile in line:
                        if tile == TileState.LION:
                            no_of_lions += 1
                shot_lions = env.scenario.LION_RATIO * 100 - no_of_lions
                records.append(RecordModel(
                    "rl_ex",
                    RecordModel.determine_result(env.scenario, oot),
                    steps_taken,
                    scenario.steps_left,
                    "extended",
                    ScenarioModel.make_parameter_string(env.scenario.TREE_RATIO, env.scenario.LION_RATIO),
                    shot_lions
                ))
                break
    db = get_instance(target_collection, 'r', "records_rl_ex")
    db.insert_many(records)

def eval_gen_extended(scenarios: list[ExtendedGame], gen_collection, target_collection):
    individuals: list[ExtendedIndividual] = get_individuals_from_collection(gen_collection, "extended", "extended_genetic_ind_actual")
    records = []
    for scenario in scenarios:
        for individual in individuals:
            scenario_copy: ExtendedGame = copy.deepcopy(scenario)
            individual.set_specific_scenario(scenario_copy)
            oot = False
            while individual.scenario.is_alive:
                individual.act()
                if individual.scenario.is_won():
                    break
                if individual.steps_made > 1000:
                    oot = True
                    break
            no_of_lions = 0
            for line in individual.scenario.board:
                for tile in line:
                    if tile == TileState.LION:
                        no_of_lions +=1
            shot_lions = individual.scenario.LION_RATIO * 100 - no_of_lions
            records.append(RecordModel(
                "genetic_ex",
                RecordModel.determine_result(individual.scenario, oot),
                individual.steps_made,
                individual.scenario.steps_left,
                "extended",
                ScenarioModel.make_parameter_string(individual.scenario.TREE_RATIO, individual.scenario.LION_RATIO),
                shot_lions
            ))
    db = get_instance(target_collection, 'r', "records_gen_ex_selected")
    db.insert_many(records)

def eval_rule_based(scenarios, target_collection):
    records = []
    for scenario_o in scenarios:
        steps = 0
        scenario = copy.deepcopy(scenario_o)
        model = RuleBasedPlayer(given_scenario=scenario)
        while model.game.is_alive:
            step = model.act()
            model.game.make_step(step)
            steps += 1
            if steps >= 500:
                break
        records.append(RecordModel(
            "rule_based",
            RecordModel.determine_result(scenario),
            steps,
            scenario.steps_left,
            "simple",
            ScenarioModel.make_parameter_string(scenario.TREE_RATIO, scenario.LION_RATIO)
        ))
    db = get_instance(target_collection, 'r', "records_rule_based")
    db.insert_many(records)

def eval_rule_based_extended(scenarios, target_collection):
    records = []
    for scenario_o in scenarios:
        steps = 0
        scenario = copy.deepcopy(scenario_o)
        model = ExtendedRuleBasePlayer(given_scenario=scenario)
        oot = False
        while model.game.is_alive:
            step = model.act()
            model.game.make_step(step)
            if step != Step.STAY:
                steps += 1
            if model.game.is_won():
                break
            if steps > 1000:
                oot = True
                break
        no_of_lions = 0
        for line in model.game.board:
            for tile in line:
                if tile == TileState.LION:
                    no_of_lions += 1
        shot_lions = model.game.LION_RATIO * 100 - no_of_lions
        records.append(RecordModel(
            "rule_based_ex",
            RecordModel.determine_result(scenario, oot),
            steps,
            scenario.steps_left,
            "extended",
            ScenarioModel.make_parameter_string(scenario.TREE_RATIO, scenario.LION_RATIO),
            shot_lions
        ))
    db = get_instance(target_collection, 'r', "records_rule_based_ex_final")
    db.insert_many(records)

#!!!!!!!!!!!!!!! always check for #changed! !!!!!!!!!!!!!!!!!!!!!!!!

# rule based records -------------------------------

# for i in range(30):
#     evaluate("simple_T40L40", "simple", "rule_based")
#     evaluate("simple_T40L30", "simple", "rule_based")
#     evaluate("simple_T30L30", "simple", "rule_based")
#     evaluate("simple_T20L40", "simple", "rule_based")
#     pass


# for i in range(30):
#     evaluate("extended_T20L10", "extended", "rule_based_ex")
#     evaluate("extended_T20L05", "extended", "rule_based_ex")
#     evaluate("extended_T25L10", "extended", "rule_based_ex")
#     evaluate("extended_T25L05", "extended", "rule_based_ex")
#     evaluate("extended_T30L05", "extended", "rule_based_ex")
#     evaluate("extended_T30L10", "extended", "rule_based_ex")
#     evaluate("extended_T30L20", "extended", "rule_based_ex")
#     evaluate("extended_T30L30", "extended", "rule_based_ex")
#     pass

# end rule based records-------------------------------



# second gen isolated -------------------------------------------------------------------

# evaluate("simple_T40L40", "simple", "gen", gen_collection="naive_simple_T40L30_10k_2-ft")
# evaluate("simple_T40L40", "simple", "gen", gen_collection="naive_simple_T40L30_5k_2-ft")
# evaluate("simple_T40L40", "simple", "gen", gen_collection="normal_simple_T40L40_5h_long-ft")
# evaluate("simple_T40L40", "simple", "gen", gen_collection="normal_simple_T40L40_5h_long_2-ft")
# evaluate("simple_T40L40", "simple", "gen", gen_collection="prog_simple_T40L40_5h_long-ft")
# evaluate("simple_T40L40", "simple", "gen", gen_collection="prog_simple_T40L40_5h_long+1k")
# evaluate("simple_T40L40", "simple", "gen", gen_collection="prog_simple_T40L40_5h_long+5k")
# evaluate("simple_T40L40", "simple", "gen", gen_collection="prog_simple_T40L40_5h_long+5k-ft")
#
# evaluate("simple_T40L30", "simple", "gen", gen_collection="naive_simple_T40L30_10k_1-ft")
# evaluate("simple_T40L30", "simple", "gen", gen_collection="naive_simple_T40L30_5k_1-ft")
# evaluate("simple_T40L30", "simple", "gen", gen_collection="normal_simple_T40L30_5h_long-ft")
# evaluate("simple_T40L30", "simple", "gen", gen_collection="prog_simple_T40L30_5h_long-ft")
# evaluate("simple_T40L30", "simple", "gen", gen_collection="prog_simple_T40L30_5h_long+1k")
# evaluate("simple_T40L30", "simple", "gen", gen_collection="prog_simple_T40L30_5h_long+5k")
# evaluate("simple_T40L30", "simple", "gen", gen_collection="prog_simple_T40L30_5h_long+5k-ft")
#
# evaluate("simple_T30L30", "simple", "gen", gen_collection="naive_simple_T30L30_10k_3-ft")
# # evaluate("simple_T30L30", "simple", "gen", gen_collection="")
# evaluate("simple_T30L30", "simple", "gen", gen_collection="normal_simple_T30L30_5h_long-ft")
# evaluate("simple_T30L30", "simple", "gen", gen_collection="prog_simple_T30L30_5h_long-ft")
# evaluate("simple_T30L30", "simple", "gen", gen_collection="prog_simple_T30L30_5h_long+1k")
# evaluate("simple_T30L30", "simple", "gen", gen_collection="prog_simple_T30L30_5h_long+5k")
# evaluate("simple_T30L30", "simple", "gen", gen_collection="prog_simple_T30L30_5h_long+5k-ft")
#
# evaluate("simple_T20L40", "simple", "gen", gen_collection="naive_simple_T20L40_10k_4-ft")
# evaluate("simple_T20L40", "simple", "gen", gen_collection="normal_simple_T20L40_5h_long-ft")
# evaluate("simple_T20L40", "simple", "gen", gen_collection="prog_simple_T20L40_5h_long-ft")
# evaluate("simple_T20L40", "simple", "gen", gen_collection="prog_simple_T20L40_5h_long+1k")
# evaluate("simple_T20L40", "simple", "gen", gen_collection="prog_simple_T20L40_5h_long+1k+1k")
# evaluate("simple_T20L40", "simple", "gen", gen_collection="prog_simple_T20L40_5h_long+1k+5k")
# evaluate("simple_T20L40", "simple", "gen", gen_collection="prog_simple_T20L40_5h_long+10k")

#further selected #dep

# evaluate("simple_T20L40", "simple", "gen", gen_collection="FURTHER-prog_simple_T20L40_5h_long-ft")
# evaluate("simple_T20L40", "simple", "gen", gen_collection="FURTHER-naive_simple_T20L40_10k_4-ft")
# evaluate("simple_T20L40", "simple", "gen", gen_collection="FURTHER-normal_simple_T20L40_5h_long-ft")
# evaluate("simple_T20L40", "simple", "gen", gen_collection="FURTHER-prog_simple_T20L40_5h_long+1k+5k")
# evaluate("simple_T20L40", "simple", "gen", gen_collection="FURTHER-prog_simple_T20L40_5h_long+10k")

# evaluate("simple_T20L40", "simple", "gen", gen_collection="FURTHER-prog_simple_T20L40_5h_long+10k-top-{x}")
# evaluate("simple_T20L40", "simple", "gen", gen_collection="FURTHER-naive_simple_T20L40_10k_4-ft-top-{x}")
# evaluate("simple_T20L40", "simple", "gen", gen_collection="FURTHER-normal_simple_T20L40_5h_long-ft-top-3")
# evaluate("simple_T20L40", "simple", "gen", gen_collection="FURTHER-prog_simple_T20L40_5h_long+1k+5k-top-3")

#dep

#further new
# evaluate("simple_T20L40", "simple", "gen", gen_collection="FURTHER-naive_simple_T20L40_10k_4-ft-top-25")
# evaluate("simple_T20L40", "simple", "gen", gen_collection="FURTHER-naive_simple_T30L30_10k_3-ft-top-25")
# evaluate("simple_T20L40", "simple", "gen", gen_collection="FURTHER-prog_simple_T20L40_5h_long+10k-top-25")
# evaluate("simple_T20L40", "simple", "gen", gen_collection="FURTHER-prog_simple_T20L40_5h_long+1k+5k-top-25")
# evaluate("simple_T20L40", "simple", "gen", gen_collection="FURTHER-prog_simple_T30L30_5h_long+5k-top-25")
# evaluate("simple_T20L40", "simple", "gen", gen_collection="FURTHER-naive_simple_T20L40_10k_4-ft-top-3")
# evaluate("simple_T20L40", "simple", "gen", gen_collection="FURTHER-naive_simple_T30L30_10k_3-ft-top-3")
# evaluate("simple_T20L40", "simple", "gen", gen_collection="FURTHER-prog_simple_T20L40_5h_long+10k-top-3")
# evaluate("simple_T20L40", "simple", "gen", gen_collection="FURTHER-prog_simple_T20L40_5h_long+1k+5k-top-3")
# evaluate("simple_T20L40", "simple", "gen", gen_collection="FURTHER-prog_simple_T30L30_5h_long+5k-top-3")
#
# evaluate("simple_T30L30", "simple", "gen", gen_collection="FURTHER-naive_simple_T20L40_10k_4-ft-top-25")
# evaluate("simple_T30L30", "simple", "gen", gen_collection="FURTHER-naive_simple_T30L30_10k_3-ft-top-25")
# evaluate("simple_T30L30", "simple", "gen", gen_collection="FURTHER-prog_simple_T20L40_5h_long+10k-top-25")
# evaluate("simple_T30L30", "simple", "gen", gen_collection="FURTHER-prog_simple_T20L40_5h_long+1k+5k-top-25")
# evaluate("simple_T30L30", "simple", "gen", gen_collection="FURTHER-prog_simple_T30L30_5h_long+5k-top-25")
# evaluate("simple_T30L30", "simple", "gen", gen_collection="FURTHER-naive_simple_T20L40_10k_4-ft-top-3")
# evaluate("simple_T30L30", "simple", "gen", gen_collection="FURTHER-naive_simple_T30L30_10k_3-ft-top-3")
# evaluate("simple_T30L30", "simple", "gen", gen_collection="FURTHER-prog_simple_T20L40_5h_long+10k-top-3")
# evaluate("simple_T30L30", "simple", "gen", gen_collection="FURTHER-prog_simple_T20L40_5h_long+1k+5k-top-3")
# evaluate("simple_T30L30", "simple", "gen", gen_collection="FURTHER-prog_simple_T30L30_5h_long+5k-top-3")



# NEW -------------

# for i in range(1,6):
    # evaluate("simple_T30L30", "simple", "gen", gen_collection=f"naive_T30L30_{i}-ft")
    # evaluate("simple_T30L30", "simple", "gen", gen_collection=f"normal_T30L30_{i}-ft")
    # evaluate("simple_T30L30", "simple", "gen", gen_collection=f"normal_1k_T30L30_{i}")
    # evaluate("simple_T30L30", "simple", "gen", gen_collection=f"prog_T30L30_{i}-ft")
    # evaluate("simple_T30L30", "simple", "gen", gen_collection=f"prog_T30L30_{i}-ft+5k")


    # evaluate("simple_T30L30", "simple", "gen", gen_collection=f"normal_T15L40_{i}-ft")

    # evaluate("simple_T30L30", "simple", "gen", gen_collection=f"naive_T20L30_{i}-ft")
    # evaluate("simple_T30L30", "simple", "gen", gen_collection=f"normal_T20L30_{i}-ft")
    # evaluate("simple_T30L30", "simple", "gen", gen_collection=f"prog_T20L30_{i}-ft")
    # evaluate("simple_T30L30", "simple", "gen", gen_collection=f"prog_T20L30_{i}-ft+5k")


    # evaluate("simple_T30L30", "simple", "gen", gen_collection=f"naive_T30L30_{i}-ft-top-10")
    # evaluate("simple_T30L30", "simple", "gen", gen_collection=f"normal_T30L30_{i}-ft-top-10")
    # evaluate("simple_T30L30", "simple", "gen", gen_collection=f"prog_T30L30_{i}-ft-top-10")
    # evaluate("simple_T30L30", "simple", "gen", gen_collection=f"prog_T30L30_{i}-ft+5k-top-10")

    # evaluate("simple_T30L30", "simple", "gen", gen_collection=f"naive_T20L30_{i}-ft-top-10")
    # evaluate("simple_T30L30", "simple", "gen", gen_collection=f"normal_T20L30_{i}-ft-top-10")
    # evaluate("simple_T30L30", "simple", "gen", gen_collection=f"prog_T20L30_{i}-ft-top-10")
    # evaluate("simple_T30L30", "simple", "gen", gen_collection=f"prog_T20L30_{i}-ft+5k-top-10")

#fixed
# evaluate("simple_T30L30", "simple", "gen", gen_collection=f"naive_T30L30_4-ft-top-10-fix-5")
# evaluate("simple_T30L30", "simple", "gen", gen_collection=f"normal_T30L30_1-ft-top-10-fix-5")
# evaluate("simple_T30L30", "simple", "gen", gen_collection=f"prog_T30L30_1-ft-top-10-fix-5")
# evaluate("simple_T30L30", "simple", "gen", gen_collection=f"prog_T30L30_1-ft+5k-top-10-fix-5")
#
# evaluate("simple_T30L30", "simple", "gen", gen_collection=f"naive_T20L30_3-ft-top-10-fix-5")
# evaluate("simple_T30L30", "simple", "gen", gen_collection=f"normal_T20L30_4-ft-top-10-fix-5")
# evaluate("simple_T30L30", "simple", "gen", gen_collection=f"prog_T20L30_4-ft-top-10-fix-5")
# evaluate("simple_T30L30", "simple", "gen", gen_collection=f"prog_T20L30_5-ft+5k-top-10-fix-5")

# evaluate("simple_T30L30", "simple", "gen", gen_collection=f"naive_T30L30_4-ft-top-10-fix-10")
# evaluate("simple_T30L30", "simple", "gen", gen_collection=f"normal_T30L30_1-ft-top-10-fix-10")
# evaluate("simple_T30L30", "simple", "gen", gen_collection=f"prog_T30L30_1-ft-top-10-fix-10")
# evaluate("simple_T30L30", "simple", "gen", gen_collection=f"prog_T30L30_1-ft+5k-top-10-fix-10")
#
# evaluate("simple_T30L30", "simple", "gen", gen_collection=f"naive_T20L30_3-ft-top-10-fix-10")
# evaluate("simple_T30L30", "simple", "gen", gen_collection=f"normal_T20L30_4-ft-top-10-fix-10")
# evaluate("simple_T30L30", "simple", "gen", gen_collection=f"prog_T20L30_4-ft-top-10-fix-10")
# evaluate("simple_T30L30", "simple", "gen", gen_collection=f"prog_T20L30_5-ft+5k-top-10-fix-10")


# end second gen isolated -------------------------------------------------------------------

# deepl isolated -------------------------------------------------------------------------------
#MUCH SLOWER TO DECIDE

# evaluate("simple_T20L40", "simple", model_type="deepl", model_path="deepl_simple_2e")
# evaluate("simple_T30L30", "simple", model_type="deepl", model_path="deepl_simple_2e")
# evaluate("simple_T40L40", "simple", model_type="deepl", model_path="deepl_simple_2e")
# evaluate("simple_T40L30", "simple", model_type="deepl", model_path="deepl_simple_2e")

# for _ in range(1):
#     evaluate("simple_T30L30", "simple", model_type="deepl", model_path="deepl_simple_short_ver2")
#     evaluate("simple_T30L30", "simple", model_type="deepl", model_path="deepl_simple_medium")
#     evaluate("simple_T30L30", "simple", model_type="deepl", model_path="deepl_simple_short")
#     evaluate("simple_T30L30", "simple", model_type="deepl", model_path="deepl_simple_very_short")

# for _ in range(3):
#     if _ > 1:
#         evaluate("simple_T30L30", "simple", model_type="deepl", model_path="MLM_very_short_1")
#         evaluate("simple_T30L30", "simple", model_type="deepl", model_path="MLM_very_short_2")
#         evaluate("simple_T30L30", "simple", model_type="deepl", model_path="MLM_very_short_3")
#         evaluate("simple_T30L30", "simple", model_type="deepl", model_path="MLM_very_short_4")
#         evaluate("simple_T30L30", "simple", model_type="deepl", model_path="MLM_very_short_5")
#
#         evaluate("simple_T30L30", "simple", model_type="deepl", model_path="MLM_short_1")
#         evaluate("simple_T30L30", "simple", model_type="deepl", model_path="MLM_short_2")
#         evaluate("simple_T30L30", "simple", model_type="deepl", model_path="MLM_short_3")
#     evaluate("simple_T30L30", "simple", model_type="deepl", model_path="MLM_short_4")
#     evaluate("simple_T30L30", "simple", model_type="deepl", model_path="MLM_short_5")
#
#     evaluate("simple_T30L30", "simple", model_type="deepl", model_path="MLM_medium_1")
#     evaluate("simple_T30L30", "simple", model_type="deepl", model_path="MLM_medium_3")
#     evaluate("simple_T30L30", "simple", model_type="deepl", model_path="MLM_medium_2")
#     evaluate("simple_T30L30", "simple", model_type="deepl", model_path="MLM_medium_4")
#     evaluate("simple_T30L30", "simple", model_type="deepl", model_path="MLM_medium_5")
#
#     evaluate("simple_T30L30", "simple", model_type="deepl", model_path="MLM_long_1")
#     evaluate("simple_T30L30", "simple", model_type="deepl", model_path="MLM_long_2")
#     evaluate("simple_T30L30", "simple", model_type="deepl", model_path="MLM_long_3")


# evaluate("simple_T20L40", "simple", model_type="deepl", model_path="deepl_simple_long")
# evaluate("simple_T30L30", "simple", model_type="deepl", model_path="deepl_simple_long")
# evaluate("simple_T40L40", "simple", model_type="deepl", model_path="deepl_simple_long")
# evaluate("simple_T40L30", "simple", model_type="deepl", model_path="deepl_simple_long")

# evaluate("simple_T20L40", "simple", model_type="deepl", model_path="deepl_simple_ml_very-long")
# evaluate("simple_T30L30", "simple", model_type="deepl", model_path="deepl_simple_ml_very-long")
# evaluate("simple_T40L40", "simple", model_type="deepl", model_path="deepl_simple_ml_very-long")
# evaluate("simple_T40L30", "simple", model_type="deepl", model_path="deepl_simple_ml_very-long")

#ml long? big long?

# evaluate("simple_T20L40", "simple", model_type="deepl", model_path="deepl_simple_big_long")
# evaluate("simple_T30L30", "simple", model_type="deepl", model_path="deepl_simple_big_long")
# evaluate("simple_T40L40", "simple", model_type="deepl", model_path="deepl_simple_big_long")
# evaluate("simple_T40L30", "simple", model_type="deepl", model_path="deepl_simple_big_long")
#
# evaluate("simple_T20L40", "simple", model_type="deepl", model_path="deepl_simple_big_short")
# evaluate("simple_T30L30", "simple", model_type="deepl", model_path="deepl_simple_big_short")
# evaluate("simple_T40L40", "simple", model_type="deepl", model_path="deepl_simple_big_short")
# evaluate("simple_T40L30", "simple", model_type="deepl", model_path="deepl_simple_big_short")
#
# evaluate("simple_T20L40", "simple", model_type="deepl", model_path="deepl_simple_big_medium")
# evaluate("simple_T30L30", "simple", model_type="deepl", model_path="deepl_simple_big_medium")
# evaluate("simple_T40L40", "simple", model_type="deepl", model_path="deepl_simple_big_medium")
# evaluate("simple_T40L30", "simple", model_type="deepl", model_path="deepl_simple_big_medium")

# for i in range(1,11):
#     path = f"SLM_medium_{i}"
#     evaluate("simple_T30L30", "simple", model_type="deepl", model_path=path)

# end deepl isolated -------------------------------------------------------------------------------

# rl isolated --------------------------------------------------------------------------------------
#runs much faster, maybe on par with gen

# for _ in range(5):
    # path = "A2C_simple_T30L30_T5S1L0"
    # evaluate("simple_T20L40", "simple", model_type="rl", model_path=path)
    # evaluate("simple_T30L30", "simple", model_type="rl", model_path=path)
    # evaluate("simple_T40L40", "simple", model_type="rl", model_path=path)
    # evaluate("simple_T40L30", "simple", model_type="rl", model_path=path)

# for _ in range(5):
#     path = "DQN_simple_T30L30_T5S1L10"
#     evaluate("simple_T20L40", "simple", model_type="rl", is_dqn=True, model_path=path)
#     evaluate("simple_T30L30", "simple", model_type="rl", is_dqn=True, model_path=path)
#     evaluate("simple_T40L40", "simple", model_type="rl", is_dqn=True, model_path=path)
#     evaluate("simple_T40L30", "simple", model_type="rl", is_dqn=True, model_path=path)


# for i in range(1,4):
#     path = "A2C_punish_T10S1L20"
#     evaluate("simple_T30L30", "simple", model_type="rl", is_dqn=False, model_path=path+ f"_{i}")
    # path = "DQN_T5S0L10_medium_5lr"
    # evaluate("simple_T30L30", "simple", model_type="rl", is_dqn=True, model_path=path + f"_{i}")
#     pass

# for i in range(1,11):
#     path = "A2C_T10S1L1000"
#     evaluate("simple_T30L30", "simple", model_type="rl", is_dqn=False, model_path=path+ f"_{i}")
#     path = "DQN_long_5xlr_T1S0L100"
#     evaluate("simple_T30L30", "simple", model_type="rl", is_dqn=True, model_path=path+ f"_{i}")
#     pass

# end rl isolated --------------------------------------------------------------------------------------

# genetic extended isolated ---------------------------------------------------------------------------


# evaluate("extended_T30L20", "extended", "gen_ex", gen_collection="naive_ex_T40L30_10k_2-ft")
# evaluate("extended_T30L20", "extended", "gen_ex", gen_collection="naive_ex_T30L30_10k_2-ft")
# evaluate("extended_T30L20", "extended", "gen_ex", gen_collection="naive_ex_T40L30_10k-ft")
# # evaluate("extended_T30L20", "extended", "gen_ex", gen_collection="naive_ex_T40L30_10k_2-ft")
#
# evaluate("extended_T30L20", "extended", "gen_ex", gen_collection="normal_ex_T30L30_5h_long-ft")
# evaluate("extended_T30L20", "extended", "gen_ex", gen_collection="normal_ex_T30L40_5h_long-ft")
# # evaluate("extended_T30L20", "extended", "gen_ex", gen_collection="normal_ex_T30L40_5h_long_2-ft")
# evaluate("extended_T30L20", "extended", "gen_ex", gen_collection="normal_ex_T40L20_5h_long-ft")
# evaluate("extended_T30L20", "extended", "gen_ex", gen_collection="normal_ex_T40L30_5h_long-ft")


# evaluate("extended_T30L20", "extended", "gen_ex", gen_collection="prog_ex_T30L30_5h_long+5k-won")
# evaluate("extended_T30L20", "extended", "gen_ex", gen_collection="prog_ex_T30L40_5h_long+5k-won")
# evaluate("extended_T30L20", "extended", "gen_ex", gen_collection="normal_ex_T30L40_5h_long_2-ft")
# evaluate("extended_T30L20", "extended", "gen_ex", gen_collection="prog_ex_T40L20_5h_long+5k-won")
# evaluate("extended_T30L20", "extended", "gen_ex", gen_collection="prog_ex_T40L30_5h_long+5k-won")


# evaluate("extended_T30L20", "extended", "gen_ex", gen_collection=f"naive_ex_T30L30_10k_12-ft")

# for i in range(1,13):
#     try:
#         # evaluate("extended_T30L20", "extended", "gen_ex", gen_collection=f"naive_ex_T30L30_10k_{i}-ft")
#         # evaluate("extended_T30L20", "extended", "gen_ex", gen_collection=f"naive_ex_T30L30_10k_{i}-ft-top-10")
#         # evaluate("extended_T30L20", "extended", "gen_ex", gen_collection=f"normal_ex_T30L20_{i}-ft-top-10")
#         # evaluate("extended_T30L20", "extended", "gen_ex", gen_collection=f"prog_ex_T30L20_{i}_real-ft-top-10")
#         # evaluate("extended_T30L20", "extended", "gen_ex", gen_collection=f"prog_ex_T30L20_{i}_real-ft+5k-top-10")
#         evaluate("extended_T30L20", "extended", "gen_ex", gen_collection=f"prog_ex_T30L20_{i}_real-ft+5k-top-10-won")
#     except TypeError:
#         print(f"{i} not found")

# evaluate("extended_T30L20", "extended", "gen_ex", gen_collection=f"prog_ex_T30L20_10_real-ft+5k-top-10-won-top-3-won")
# evaluate("extended_T30L20", "extended", "gen_ex", gen_collection=f"prog_ex_T30L20_10_real-ft+5k-top-10-won")
    # evaluate("extended_T30L20", "extended", "gen_ex", gen_collection=f"naive_ex_T30L20_10k_{i}-ft")
    # evaluate("extended_T30L20", "extended", "gen_ex", gen_collection=f"normal_ex_T30L20_{i}-ft")
    # evaluate("extended_T30L20", "extended", "gen_ex", gen_collection=f"prog_ex_T30L20_{i}-ft")
    # evaluate("extended_T30L20", "extended", "gen_ex", gen_collection=f"prog_ex_T30L20_{i}_real-ft")
    # evaluate("extended_T30L20", "extended", "gen_ex", gen_collection=f"prog_ex_T30L20_{i}_real-ft+5k")


# end genetic extended isolated ---------------------------------------------------------------------------


# deepl extended isolated -----------------------------------------------------------------------------------



# for i in range(2,4):
#     path = "MLM_ex_long"
#     evaluate("extended_T30L20", "extended", "deepl_ex", model_path=path+f"_{i}")



# for i in range(1, 6):
#     evaluate("extended_T30L20", "extended", "deepl_ex", model_path=f"MLM_ex_very_short_{i}")


# end deepl extended isolated -----------------------------------------------------------------------------------

# rl extended isolated --------------------------------------------------------------------------------------------


# for i in range(5,6):
#     evaluate("extended_T30L20", "extended", "rl_ex", model_path=f"A2C_ex_T1S0L0K2_{i}")
#     evaluate("extended_T30L20", "extended", "rl_ex", is_dqn=True, model_path=f"DQN_ex_medium_T1S0L0K2_{i}")
#     evaluate("extended_T30L20", "extended", "rl_ex", is_dqn=True, model_path=f"DQN_ex_long_T1S0L0K2_{i}")




# end rl extended isolated --------------------------------------------------------------------------------------------

#
#generated scenario bundles
#
# generate_scenarios(0.3,0.3,"simple","simple_T30L30","scenarios_gen", num_of_scenarios=20)
# generate_scenarios(0.2,0.4,"simple","simple_T20L40","scenarios_gen")
# generate_scenarios(0.25,0.4,"simple","simple_T25L40","scenarios_gen", num_of_scenarios=30)
#
# generate_scenarios(0.3, 0.3, "extended", "extended_T30L30", "scenarios_ex_2", num_of_scenarios=30)
# generate_scenarios(0.3, 0.2, "extended", "extended_T30L20", "scenarios_ex_2", num_of_scenarios=30)
# generate_scenarios(0.3, 0.1, "extended", "extended_T30L10", "scenarios_ex_2", num_of_scenarios=30)
# generate_scenarios(0.3, 0.05, "extended", "extended_T30L05", "scenarios_ex_2", num_of_scenarios=30)
# generate_scenarios(0.25, 0.1, "extended", "extended_T25L10", "scenarios_ex_2", num_of_scenarios=30)
# generate_scenarios(0.25, 0.05, "extended", "extended_T25L05", "scenarios_ex_2", num_of_scenarios=30)
# generate_scenarios(0.2, 0.1, "extended", "extended_T20L10", "scenarios_ex_2", num_of_scenarios=30)
# generate_scenarios(0.2, 0.05, "extended", "extended_T20L05", "scenarios_ex_2", num_of_scenarios=30)

#
# generate_scenarios(0.3, 0, "simple", "simple_T30L00")
# generate_scenarios(0.3, 0.1, "simple", "simple_T30L10")
# generate_scenarios(0.3, 0.3, "simple", "simple_T30L30")
# generate_scenarios(0.3, 0.4, "simple", "simple_T30L40")
# generate_scenarios(0.3, 0.5, "simple", "simple_T30L50")
# generate_scenarios(0.2, 0.4, "simple", "simple_T30L40")
# generate_scenarios(0.2, 0.6, "simple", "simple_T30L60")
# generate_scenarios(0.15, 0.4, "simple", "simple_T15L40")
# generate_scenarios(0.1, 0.4, "simple", "simple_T10L40")
# generate_scenarios(0.3, 0.1, "extended", "extended_T30L10")
# generate_scenarios(0.3, 0.2, "extended", "extended_T30L20")
# generate_scenarios(0.3, 0.3, "extended", "extended_T30L30")
# generate_scenarios(0.3, 0.4, "extended", "extended_T30L40")
# generate_scenarios(0.4, 0.4, "extended", "extended_T40L40")
# generate_scenarios(0.4, 0.5, "extended", "extended_T40L50")
# generate_scenarios(0.3, 0.5, "extended", "extended_T30L50")
# generate_scenarios(0.2, 0.3, "extended", "extended_T20L30")
# generate_scenarios(0.2, 0.4, "extended", "extended_T20L40")
# generate_scenarios(0.15, 0.3, "extended", "extended_T15L30")
# generate_scenarios(0.15, 0.4, "extended", "extended_T15L40")
# generate_scenarios(0.10, 0.4, "extended", "extended_T10L40")
