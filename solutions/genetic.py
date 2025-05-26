# fitness function, the number of steps the individual managed to take
# reproduction? crossover x% of the time
# mutation? make them forget steps? change step randomly, crossover x% of the time
# selection? based on fitness function, keep individual if their fitness is over a certain percentile of all individuals(top x%)
import copy

from game_environment.scenario import SimpleGame, ContextHolder, ContextBasedGame, ExtendedGame
from utils.scenario_utils import Environment, Step, ExtendedStep, TileState, ResultOfStep, ExtendedResultOfStep
from random import choice, random
from utils.db_context import get_instance
from utils.db_entities import GeneticIndividualModel
from utils.genetic_action_utils import ActionHolder, Action

POPULATION_SIZE = 10000

CROSSOVER_RATE = 0.7
MUTATION_RATE = 0.05


# two approaches I reckon
# First: offspring inherits all moves, forget some, missing spots are filled with completely new individuals
# Second: surviving individuals are shuffled to create offspring, crossover, missing spots are filled via randomly selecting moves from parent generation?

def models_to_individuals(models: list[GeneticIndividualModel]):
    individuals = []
    for model in models:
        individuals.append(Individual(
            model.ind_id.split(',')[0],
            model.ind_id.split(',')[1],
            known_actions=model.action_set.actions,
            parent_id=model.parent_id,
            other_parent_id=model.other_parent_id
        ))
    return individuals

def save_individuals(col, individuals_to_save, env_type):
    if len(individuals_to_save) == 0:
        return
    # for individual in individuals_to_save:
    #     for a in individual.known_actions.actions:
    #         if a.env.get_as_list()[a.step.value] == 2:
    #             print(f"bad step {a.env.get_as_list()} - {a.step.value} before save")
    db = get_instance(col, 'g', "extended_genetic_ind_actual") #changed!
    models = []
    for ind in individuals_to_save:
        models.append(
            GeneticIndividualModel(
                ind.id,
                ind.known_actions,
                env_type,
                ind.parent_id,
                ind.other_parent_id
            )
        )
    db.insert_many(models)

class Individual:

    def __init__(self, generation: int, seq_num: int,
                 known_actions: list[Action] = None,
                 parent_id: str = None, other_parent_id: str = None,
                 game_tree_ratio: float = None, game_lion_ratio: float = None):
        self.generation = generation
        self.id = f"{self.generation},{seq_num}"
        self.parent_id = parent_id
        self.other_parent_id = other_parent_id
        if known_actions is not None:
            self.known_actions = ActionHolder(copy.deepcopy(known_actions))
        else:
            self.known_actions = ActionHolder()
        if game_tree_ratio is not None:
            self.scenario = SimpleGame(tree_ratio=game_tree_ratio, lion_ratio=game_lion_ratio)
        else:
            self.scenario = SimpleGame()
        self.steps_made = 0
        self.previous_step = None #roundabout solution

    def act(self) -> ResultOfStep:
        current_environment = self.scenario.get_environment()
        if self.known_actions.is_env_known(current_environment):
            step_to_take = self.known_actions.get_action_for_env(current_environment).step
        else:
            step_to_take = choice((Step.UP, Step.RIGHT, Step.DOWN, Step.LEFT))
            self.known_actions.add_action(Action(current_environment, step_to_take))
        self.steps_made += 1
        return self.scenario.make_step(step_to_take)

        # current_environment = self.scenario.get_environment()
        # if self.known_actions.is_env_known(current_environment):
        #     step_to_take = self.known_actions.get_action_for_env(current_environment).step
        # else:
        #     step_to_take = choice((Step.UP, Step.RIGHT, Step.DOWN, Step.LEFT))
        # self.scenario.make_step(step_to_take)
        # if not self.known_actions.is_env_known(current_environment) and self.scenario.is_alive:
        #     self.known_actions.add_action(Action(current_environment, step_to_take))

    def set_specific_scenario(self, scenario):
        """scenario must be an initialized game"""
        self.scenario = scenario
        self.steps_made = 0

    def is_fully_trained(self):
        return len(self.known_actions.actions) >= 80

class ExtendedIndividual(Individual):

    def __init__(self, generation: int, seq_num: int,
                 known_actions: list[Action] = None,
                 parent_id: str = None, other_parent_id: str = None,
                 game_tree_ratio: float = None, game_lion_ratio: float = None):
        super().__init__(generation, seq_num, copy.deepcopy(known_actions), parent_id, other_parent_id)
        if game_tree_ratio is not None:
            self.scenario = ExtendedGame(tree_ratio=game_tree_ratio, lion_ratio=game_lion_ratio)
        else:
            self.scenario = ExtendedGame()

    def act(self) -> ResultOfStep:
        current_environment = self.scenario.get_environment()
        if self.known_actions.is_env_known(current_environment):
            step_to_take = self.known_actions.get_action_for_env(current_environment).step
        else:
            step_to_take = choice((Step.UP, Step.RIGHT, Step.DOWN, Step.LEFT, Step.STAY))
            self.known_actions.add_action(Action(current_environment, step_to_take))
        if step_to_take != Step.STAY:
            self.steps_made += 1
        self.previous_step = step_to_take
        return self.scenario.make_step(step_to_take)

    def is_fully_trained(self):
        return len(self.known_actions.actions) == 81


class GeneticNaive:

    def __init__(self, number_of_individuals: int = POPULATION_SIZE,
                 existing_generation = None, #for further testing
                 game_tree_ratio: float = None, game_lion_ratio: float = None, # for specifying the environment ratios
                 individual_type: Individual = Individual):
        self.game_tree_ratio = game_tree_ratio
        self.game_lion_ratio = game_lion_ratio
        self.individual_type = individual_type
        self.generation = self._initialize_generation(existing_generation, number_of_individuals)
        self.number_of_individuals = number_of_individuals

    def train(self, cycles: int, do_save: bool = False, target_collection: str = None):
        fully_trained_individuals = []
        inds_with_won_games = []
        cycle = 1

        while cycle <= cycles:
            survivors = []
            # print(f"cycle {cycle} started")

            for individual in self.generation:
                individual.act()

            print(len(self.generation))


            for individual in self.generation:
                if individual.scenario.is_alive:
                    if individual.is_fully_trained():
                        fully_trained_individuals.append(individual)
                    else:
                        survivors.append(individual)

            children = []
            for individual in survivors:
                if individual.scenario.is_won():
                    inds_with_won_games.append(individual)
                    individual.scenario.is_alive = False
                if (individual.steps_made % 3 == 0
                        and individual.previous_step != Step.STAY):
                        #and len(children) < self.number_of_individuals - len(survivors)):
                    new_ind = self.individual_type(cycle, len(children),
                                                known_actions=individual.known_actions.actions,
                                                parent_id=individual.id,
                                                game_tree_ratio=self.game_tree_ratio,
                                                game_lion_ratio=self.game_lion_ratio)
                    children.append(new_ind)
            survivors.extend(children)
            self.generation = survivors
            # print(f"cycle: {cycle}, num of survivors: {len(survivors)}")

            if len(survivors) == 0:
                break
            cycle += 1
        print("training finished")
        if do_save:
            save_individuals(target_collection, self.generation, "naive")
            save_individuals(target_collection+"-ft", fully_trained_individuals, "naive")
            save_individuals(target_collection+"won", inds_with_won_games, 'naive')

    def _initialize_generation(self, existing_generation, population_size):
        if existing_generation is not None:
            return existing_generation
        else:
            return [self.individual_type(generation=0, seq_num=i, game_tree_ratio=self.game_tree_ratio, game_lion_ratio=self.game_lion_ratio) for i in range(population_size)]

    def _create_new_individual(self, generation, seq_num, parent):
        return self.individual_type(generation, seq_num,
                                                known_actions=parent.known_actions.actions,
                                                parent_id=parent.id,
                                                game_tree_ratio=self.game_tree_ratio,
                                                game_lion_ratio=self.game_lion_ratio)

class Genetic:

    def __init__(self, number_of_individuals: int = POPULATION_SIZE,
                 existing_generation=None,  # for further training or testing
                 game_tree_ratio: float = None, game_lion_ratio: float = None,  # for specifying the environment ratios
                 individual_type: Individual = Individual):
        self.game_tree_ratio = game_tree_ratio
        self.game_lion_ratio = game_lion_ratio
        self.individual_type = individual_type
        self.number_of_individuals = number_of_individuals
        self.generation = self._initialize_generation(existing_generation, number_of_individuals)
        self.newest_generation_number = 0

    def train(self, cycles: int, do_save: bool = False, target_collection: str = None):
        print("starting genetic")
        cycle = 1
        fully_trained_individuals = []
        inds_with_won_games = []
        while cycle <= cycles:
            print(f"cycle {cycle} started")
            self.newest_generation_number = cycle
            for individual in self.generation:
                individual.act()
                if individual.scenario.is_won() and individual.scenario.is_alive:
                    inds_with_won_games.append(individual)
                    individual.scenario.is_alive = False
                if individual.is_fully_trained() and individual.scenario.is_alive:
                    fully_trained_individuals.append(individual)
                    individual.scenario.is_alive = False # messes up progressive, but otherwise is necessary
                    pass
            self.selection()
            print(f"{len(self.generation)} individuals selected")
            if len(self.generation) == 0:
                break
            self.reproduction()
            if cycle < cycles - 5:
                self.mutation()
            cycle += 1

        if do_save:
            save_individuals(target_collection, self.generation, "normal")
            save_individuals(target_collection+"-ft", fully_trained_individuals, "normal")
            save_individuals(target_collection+"-won", inds_with_won_games, "normal")
        print("end genetic")

    def selection(self):
        selected = []
        for individual in self.generation:
            if individual.scenario.is_alive:
                selected.append(individual)
            # if Genetic.fitness(individual):
            #     selected.append(individual)
        self.generation = selected

    def mutation(self):
        for individual in self.generation:
            if individual.generation == self.newest_generation_number and random() < MUTATION_RATE:
                #forget a move
                individual.known_actions.remove_action(choice(individual.known_actions.actions))

    def reproduction(self):
        # can 'newborns' be parents?
        # I say yes, bc we get more varied children that way
        seq_num = 0 # for db
        while len(self.generation) < self.number_of_individuals and len(self.generation) != 0:
            #pick parents
            parent1 = choice(self.generation)
            parent2 = choice(self.generation)
            #create new moving function set
            inherited_actions: ActionHolder = ActionHolder()
            # if random() < CROSSOVER_RATE:
            #     parent1, parent2 = parent2, parent1
            for action in parent1.known_actions.actions:
                inherited_actions.add_action(action)
            for action in parent2.known_actions.actions:
                inherited_actions.add_action(action)
            #create child
            new_individual = self.individual_type(self.newest_generation_number, seq_num,
                                                           copy.deepcopy(inherited_actions.actions),
                                                           parent1.id, parent2.id,
                                                           self.game_tree_ratio, self.game_lion_ratio)
            seq_num += 1
            self.generation.append(new_individual)

    @staticmethod
    def fitness(individual: Individual) -> bool:
        #should we check learnt moves or just based on steps? OR just if its alive?
        #maybe other factors kick in after a certain number of gens
        #top 50%?
        for a in individual.known_actions.actions:
            if a.env.get_as_list()[a.step.value] == 2:
                return False
        return individual.scenario.is_alive

    def _initialize_generation(self, existing_generation, population_size, scenario_type: SimpleGame = ExtendedGame): #change scenario_type
        if existing_generation is not None:
            for ind in existing_generation:
                ind.set_specific_scenario(scenario_type(tree_ratio=self.game_tree_ratio, lion_ratio=self.game_lion_ratio))
            return existing_generation
        else:
            return [self.individual_type(0, _, game_tree_ratio=self.game_tree_ratio, game_lion_ratio=self.game_lion_ratio) for _ in range(population_size)]


####################################################################
#DEP deprecated most likely
#####################################################################

class ActionWithContext:

    def __init__(self, env: Environment, step: Step, context: ContextHolder):
        self.env = env
        self.step = step
        self.context = context

    def env_eq(self, other):
        return self.env == other.env

    def context_eq(self, other):
        return ContextHolder.contexts_eq(self.context.get_context(), other.context.get_context())

    def env_and_context_eq(self, other):
        return self.env_eq(other) and self.context_eq(other)

    def __eq__(self, other):
        return self.step == other.step and self.env_and_context_eq(other) ##dep, maybe refactor eqs to static

class ActionWithContextHolder:
    #DEP not happy with this, refactor if possible

    def __init__(self, actions: list[ActionWithContext] = None):
        if actions is None:
            actions = []
        self.actions = actions

    def add_action(self, action: ActionWithContext):
        if self.is_env_known(action.env, action.context):
            return
        self.actions.append(action)

    def remove_action(self, action: ActionWithContext):
        try:
            self.actions.remove(action)
        except ValueError:
            raise Exception("tried to delete non-existent action")

    def is_env_known(self, env: Environment, context: ContextHolder) -> bool:
        """modified to also consider context"""
        known = False
        for act in self.actions:
            if act.env == env and ContextHolder.contexts_eq(act.context.get_context(), context.get_context()):
                known = True
        return known

    def get_action_for_env(self, env: Environment, context: ContextHolder) -> ActionWithContext:
        """modified to also consider context"""
        action = None
        for act in self.actions:
            if act.env == env and ContextHolder.contexts_eq(act.context.get_context(), context.get_context()):
                action = act
        if action is not None:
            return action
        raise Exception("action not known")


class IndividualWithContext:
    #DEP test, refactor maybe

    def __init__(self, generation: int, known_actions: ActionWithContextHolder = ActionWithContextHolder(),  scenario: ContextBasedGame  = ContextBasedGame):
        self.generation = generation
        self.known_actions = known_actions
        self.scenario = scenario.__init__() #DEP this is not gonna be okay, if i wanna give them predefined scenarios!!!!!!
        self.steps_made = 0

    def act(self):
        current_environment = self.scenario.get_environment()
        current_context = self.scenario.get_context()
        if self.known_actions.is_env_known(current_environment, current_context):
            step_to_take = self.known_actions.get_action_for_env(current_environment, current_context)
        else:
            step_to_take = choice((Step.UP, Step.RIGHT, Step.DOWN, Step.LEFT))
            self.known_actions.add_action(ActionWithContext(current_environment, step_to_take, current_context))
        self.steps_made += 1
        self.scenario.make_step(step_to_take)


class GeneticNaiveWithContext:
    #dep check if child should inherit food amount
    def __init__(self, number_of_individuals: int = POPULATION_SIZE):
        self.generation = [IndividualWithContext(0) for _ in range(number_of_individuals)]

    def train(self, cycles: int):
        #dep, since fully training is unreasonable, what do we do?
        cycle = 1 #for logging
        while cycle <= cycles:
            survivors = []
            print(f"cycle {cycle} started")
            # advance every individual
            for individual in [_ for _ in self.generation]: #??, necessary? to make copy I guess
                individual.act()
                if individual.env.is_alive:
                    survivors.append(individual)
                    #offspring
                    if individual.steps_made % 3 == 0:
                        new_ind = IndividualWithContext(cycle, individual.known_actions)
                        survivors.append(new_ind)

            self.generation = survivors
            print(f"cycle: {cycle}, num of survivors: {len(survivors)}")

            if len(survivors) == 0:
                break
            cycle += 1
        #dep store stuff
        print("training finished")

class GeneticWithContext:

    def __init__(self, number_of_individuals: int = POPULATION_SIZE):
        self.generation = [IndividualWithContext(0) for _ in range(number_of_individuals)]
        self.newest_generation_number = 0

    def train(self, cycles: int):
        print("starting genetic")
        cycle = 1
        while cycle <= cycles:
            print(f"cycle {cycle} started")
            self.newest_generation_number = cycle
            for individual in self.generation:
                individual.act()

            self.selection()
            print(f"{len(self.generation)} individuals selected")
            self.reproduction()
            self.mutation()
        #dep stats
        print("end genetic")

    def selection(self):
        selected = []
        for individual in self.generation:
            if self.fitness(individual):
                selected.append(individual)
        self.generation = selected

    def mutation(self):
        #dep test
        for individual in self.generation:
            if individual.generation == self.newest_generation_number and random() < MUTATION_RATE:
                #forget a move
                individual.known_actions.remove_action(choice(individual.known_actions.actions))

    def reproduction(self):
        while len(self.generation) < POPULATION_SIZE:
            #pick parents
            parent1 = choice(self.generation)
            parent2 = choice(self.generation)
            #create new moving function set
            inherited_actions: ActionWithContextHolder = ActionWithContextHolder()
            if random() < CROSSOVER_RATE: #kinda dumb
                parent1, parent2 = parent2, parent1
            for action in parent1.known_actions.actions:
                inherited_actions.add_action(action)
            for action in parent2.known_actions.actions:
                inherited_actions.add_action(action)
            #create child
            new_individual = IndividualWithContext(self.newest_generation_number, inherited_actions)
            #add child to generation(population)
            self.generation.append(new_individual)

    @staticmethod
    def fitness(individual: IndividualWithContext) -> bool:
        #should we check learnt moves or just based on steps? OR just if its alive?
        #maybe other factors kick in after a certain number of gens
        if not individual.scenario.is_alive:
            return False
        #dep otherwise, check other parameters
        return True