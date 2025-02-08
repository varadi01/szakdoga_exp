#gets set of scenarios, and evaluates solutions on each
import copy

import numpy as np

from utils.db_context import get_instance
from utils.scenario_utils import Position, TileState, Step
from game_environment.scenario import SimpleGame, ExtendedGame
from utils.db_entities import ScenarioModel, GeneticIndividualModel, RecordModel
from solutions.genetic import Individual, GeneticNaive, Genetic, ExtendedIndividual
from solutions.rl import CustomEnvForSimpleGame, CustomEnvForExtendedGame, Agent
from solutions.deepl import load_model, Deepl, ExtendedDeepl
from utils.genetic_action_utils import ActionHolder

def get_scenarios_from_collection(collection_name: str, env_type: str):
    db = get_instance(collection_name, 's')
    scs: list[ScenarioModel] = db.get_all()
    games = []
    if env_type == "simple":
        for s in scs:
            games.append(SimpleGame(
                board= construct_board(s.board),
                spawn= Position(s.spawn[0], s.spawn[0])
            ))
        print(f"found {len(games)} scenarios")
    elif env_type == "extended":
        for s in scs:
            games.append(ExtendedGame(
                board= construct_board(s.board),
                spawn= Position(s.spawn[0], s.spawn[0])
            ))
        print(f"found {len(games)} scenarios")
    return games

def get_individuals_from_collection(collection_name: str, env_type: str):
    db = get_instance(collection_name, 'g')
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
    return individuals

def generate_scenarios(tree_ratio, lion_ratio, env_type, collection_name, num_of_scenarios = 10):
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
        db = get_instance(collection_name, 's')
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
        db = get_instance(collection_name, 's')
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

#TODO ex

def evaluate(collection: str, env_type: str, model_type: str, target_collection: str, model_path: str = None, gen_collection: str = None):
    scenarios = get_scenarios_from_collection(collection, env_type)
    #loop this? more times
    match model_type:
        case "deepl":
            eval_deepl(scenarios, model_path, target_collection)
        case "rl":
            eval_rl(scenarios, model_path, target_collection)
        case "gen":
            eval_gen(scenarios, gen_collection, target_collection)
        case "deepl-ex":
            eval_deepl_extended(scenarios, model_path, target_collection)
        case "rl-ex":
            eval_rl_extended(scenarios, model_path, target_collection)
        case "gen-ex":
            eval_gen_extended(scenarios, gen_collection, target_collection)


def eval_deepl(scenarios, model_path, target_collection):
    model = load_model(model_path)
    records = []
    for scenario_o in scenarios:
        steps = 0
        scenario = copy.deepcopy(scenario_o)
        while scenario.is_alive:
            env = scenario.get_environment()
            prediction = model.predict(np.array(env.get_as_list())[None, ...])
            step_int = np.argmax(prediction)
            scenario.make_step(Step(step_int))
            steps += 1
        records.append(RecordModel(
            "deepl",
            RecordModel.determine_result(scenario.is_alive, scenario.steps_left),
            steps,
            scenario.steps_left,
            "simple",
            ScenarioModel.make_parameter_string(scenario.TREE_RATIO, scenario.LION_RATIO)
        ))
    db = get_instance(target_collection, 'r')
    db.insert_many(records)

def eval_rl(scenarios, model_path, target_collection):
    agent = Agent(name="eval")
    agent.load_model(model_path, SimpleGame)
    records = []
    for scenario_o in scenarios:
        scenario = copy.deepcopy(scenario_o) #maybe dont need
        env = CustomEnvForSimpleGame(scenario, True)
        obs, _ = env.reset()
        steps_taken = 0
        while True: #maybe is_alive, check
            action, _states = agent.model.predict(obs)
            obs, rewards, done, _, info = env.step(action)
            steps_taken += 1
            if done:
                records.append(RecordModel(
                    "rl",
                    RecordModel.determine_result(env.scenario.is_alive, env.scenario.steps_left),
                    steps_taken,
                    scenario.steps_left,
                    "simple",
                    ScenarioModel.make_parameter_string(env.scenario.TREE_RATIO, env.scenario.LION_RATIO)
                ))
                break
    db = get_instance(target_collection, 'r')
    db.insert_many(records)

def eval_gen(scenarios: list[SimpleGame], gen_collection, target_collection):
    individuals: list[Individual] = get_individuals_from_collection(gen_collection, "simple")
    records = []
    for scenario in scenarios:
        for individual in individuals:
            scenario_copy: SimpleGame = copy.deepcopy(scenario)
            individual.set_specific_scenario(scenario_copy)
            while individual.scenario.is_alive:
                individual.act()

            records.append(RecordModel(
                "genetic",
                RecordModel.determine_result(individual.scenario.is_alive, individual.scenario.steps_left),
                individual.steps_made,
                individual.scenario.steps_left,
                "simple",
                ScenarioModel.make_parameter_string(individual.scenario.TREE_RATIO, individual.scenario.LION_RATIO)
            ))
    db = get_instance(target_collection, 'r')
    db.insert_many(records)


def eval_deepl_extended(scenarios, model_path, target_collection):
    model = load_model(model_path)
    records = []
    for scenario_o in scenarios:
        steps = 0
        scenario = copy.deepcopy(scenario_o)
        while scenario.is_alive:
            env = scenario.get_environment()
            prediction = model.predict(np.array(env.get_as_list())[None, ...])
            step_int = np.argmax(prediction)
            scenario.make_step(Step(step_int))
            steps += 1
        records.append(RecordModel(
            "deepl_ex",
            RecordModel.determine_result(scenario.is_alive, scenario.steps_left),
            steps,
            scenario.steps_left,
            "extended",
            ScenarioModel.make_parameter_string(scenario.TREE_RATIO, scenario.LION_RATIO)
        ))
    db = get_instance(target_collection, 'r')
    db.insert_many(records)

def eval_rl_extended(scenarios, model_path, target_collection):
    agent = Agent(name="eval")
    agent.load_model(model_path, SimpleGame)
    records = []
    for scenario_o in scenarios:
        scenario = copy.deepcopy(scenario_o) #maybe dont need
        env = CustomEnvForExtendedGame(scenario, True)
        obs, _ = env.reset()
        steps_taken = 0
        while True: #maybe is_alive, check
            action, _states = agent.model.predict(obs)
            obs, rewards, done, _, info = env.step(action)
            steps_taken += 1
            if done:
                records.append(RecordModel(
                    "rl_ex",
                    RecordModel.determine_result(env.scenario.is_alive, env.scenario.steps_left),
                    steps_taken,
                    scenario.steps_left,
                    "extended",
                    ScenarioModel.make_parameter_string(env.scenario.TREE_RATIO, env.scenario.LION_RATIO)
                ))
                break
    db = get_instance(target_collection, 'r')
    db.insert_many(records)

def eval_gen_extended(scenarios: list[ExtendedGame], gen_collection, target_collection):
    individuals: list[ExtendedIndividual] = get_individuals_from_collection(gen_collection, "extended")
    records = []
    for scenario in scenarios:
        for individual in individuals:
            scenario_copy: ExtendedGame = copy.deepcopy(scenario)
            individual.set_specific_scenario(scenario_copy)
            while individual.scenario.is_alive:
                individual.act()

            records.append(RecordModel(
                "genetic_ex",
                RecordModel.determine_result(individual.scenario.is_alive, individual.scenario.steps_left),
                individual.steps_made,
                individual.scenario.steps_left,
                "extended",
                ScenarioModel.make_parameter_string(individual.scenario.TREE_RATIO, individual.scenario.LION_RATIO)
            ))
    db = get_instance(target_collection, 'r')
    db.insert_many(records)


#generated scenario bundles
# 0.5, 0, "simple", "simple_T50L00"
# 0.3, 0, "simple", "simple_T30L00"
# 0.3, 0.1, "simple", "simple_T30L10"
# 0.3, 0.3, "simple", "simple_T30L30"
# 0.4, 0.4, "simple", "simple_T40L40"
# 0.3, 0.5, "simple", "simple_T30L50"
# 0.3, 0.1, "extended", "extended_T30L10"
# 0.3, 0.3, "extended", "extended_T30L30"
# 0.4, 0.4, "extended", "extended_T40L40"
# 0.3, 0.5, "extended", "extended_T30L50"
# generate_scenarios(0.3, 0.1, "extended", "extended_T30L10")
# generate_scenarios(0.3, 0.3, "extended", "extended_T30L30")
# generate_scenarios(0.4, 0.4, "extended", "extended_T40L40")
# generate_scenarios(0.3, 0.5, "extended", "extended_T30L50")
