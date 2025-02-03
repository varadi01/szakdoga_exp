#gets set of scenarios, and evaluates solutions on each

import numpy as np

from utils.db_context import get_instance
from utils.scenario_utils import Position, TileState, Step
from game_environment.scenario import SimpleGame
from utils.db_entities import ScenarioModel, GeneticIndividualModel
from solutions.genetic import Individual, GeneticNaive, Genetic
from solutions.rl import CustomEnvForSimpleGame, Agent
from utils.genetic_action_utils import ActionHolder

def get_scenarios_from_collection(collection_name: str):
    db = get_instance(collection_name, 's')
    scs: list[ScenarioModel] = db.get_all()
    games = []
    for s in scs:
        games.append(SimpleGame(
            board= construct_board(s.board),
            spawn= Position(s.spawn[0], s.spawn[0])
        ))
    print(f"found {len(games)} scenarios")
    return games

def get_individuals_from_collection(collection_name: str):
    db = get_instance(collection_name, 'g')
    ind_models = db.get_all()
    individuals = []
    for model in ind_models:
        individuals.append(Individual(
            model.ind_id.split(',')[0],
            model.ind_id.split(',')[1],
            known_actions=ActionHolder(model.action_set),
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


def evaluate(collection: str, model_type: str, model_path: str, gen_collection: str):
    scenarios = get_scenarios_from_collection(collection)
    match model_type:
        case "deepl":
            eval_deepl(scenarios, model_path)
        case "rl":
            eval_rl(scenarios, model_path)
        case "gen":
            eval_gen(scenarios, gen_collection)
        case "deepl-ex":
            pass
        case "rl-ex":
            pass
        case "gen-ex":
            pass


def eval_deepl(scenarios, model_path):
    #todo import model
    model = "todo"
    for scenario in scenarios:
        steps = 0
        while scenario.is_alive:
            env = scenario.get_environment()
            prediction = model.predict(np.array(env.get_as_list())[None, ...])
            step_int = np.argmax(prediction)
            scenario.make_step(Step(step_int))
            steps += 1
        #todo record

def eval_rl(scenarios, model_path):
    agent = Agent(name="eval")
    agent.load_model(model_path, SimpleGame)
    for scenario in scenarios:
        env = CustomEnvForSimpleGame(scenario, True)
        obs, _ = env.reset()
        while True: #maybe is_alive, check
            action, _states = agent.model.predict(obs)
            obs, rewards, done, _, info = env.step(action)
            if done:
                # todo record
                break

def eval_gen(scenarios: list[SimpleGame], gen_collection):
    eval_tries = 10 # ten times on each scenario?
    individuals: list[Individual] = get_individuals_from_collection(gen_collection)
    for i in range(eval_tries):
        for scenario in scenarios:
            for individual in individuals:
                scenario_copy: SimpleGame = scenario #clone?
                individual.set_specific_scenario(scenario_copy)
                while individual.scenario.is_alive:
                    individual.act()
                steps_taken = individual.steps_made
                food_left = individual.scenario.steps_left
                #todo record
