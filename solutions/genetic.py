# fitness function, the number of steps the individual managed to take
# reproduction? crossover x% of the time
# mutation? make them forget steps? change step randomly, crossover x% of the time
# selection? based on fitness function, keep individual if their fitness is over a certain percentile of all individuals(top x%)

from game_environment.scenario import SimpleGame, ContextHolder, ContextBasedGame
from utils.scenario_utils import Environment, Step
from random import choice, random
from utils.db_context import get_instance
from utils.db_entities import GeneticIndividualModel
from utils.genetic_action_utils import ActionHolder, Action

POPULATION_SIZE = 20000

CROSSOVER_RATE = 0.7
MUTATION_RATE = 0.05


# two approaches I reckon
# First: offspring inherits all moves, forget some, missing spots are filled with completely new individuals
# Second: surviving individuals are shuffled to create offspring, crossover, missing spots are filled via randomly selecting moves from parent generation?


class Individual:

    def __init__(self, generation: int, seq_num: int,
                 known_actions: ActionHolder = ActionHolder(),
                 parent_id: str = None, other_parent_id: str = None,
                 scenario = None):
        self.generation = generation
        self.id = f"{self.generation},{seq_num}"
        self.parent_id = parent_id
        self.other_parent_id = other_parent_id
        self.known_actions = known_actions
        if scenario is not None:
            self.scenario = scenario
        else:
            self.scenario = SimpleGame()
        self.steps_made = 0

    def act(self):
        current_environment = self.scenario.get_environment()
        if self.known_actions.is_env_known(current_environment):
            step_to_take = self.known_actions.get_action_for_env(current_environment)
        else:
            step_to_take = choice((Step.UP, Step.RIGHT, Step.DOWN, Step.LEFT))
            self.known_actions.add_action(Action(current_environment, step_to_take))
        self.steps_made += 1
        self.scenario.make_step(step_to_take)

    def set_scenario(self, game):
        self.scenario = game
        self.steps_made = 0


class GeneticNaive:
    #TODO check if child should inherit food amount
    def __init__(self, number_of_individuals: int = POPULATION_SIZE, existing_generation = None, given_scenario = None):
        if existing_generation is not None:
            self.generation = existing_generation
        else:
            self.generation = [Individual(0, _) for _ in range(number_of_individuals)]
        self.given_scenario = given_scenario

    def train(self, cycles: int, do_save: bool = False, target_collection: str = None):
        fully_trained_individuals = []
        cycle = 1 #for logging
        while cycle <= cycles:
            survivors = []
            print(f"cycle {cycle} started")
            # advance every individual
            for individual in [_ for _ in self.generation]: #??, necessary? to make copy I guess
                individual.act()
                if individual.scenario.is_alive:
                    survivors.append(individual)
                    #offspring
                    if individual.steps_made % 3 == 0:
                        if self.given_scenario is not None: #csunya
                            new_ind = Individual(cycle, len(survivors),
                                                    known_actions=individual.known_actions,
                                                    parent_id=individual.id,
                                                    scenario=self.given_scenario())
                        else:
                            new_ind = Individual(cycle, len(survivors),
                                                 known_actions=individual.known_actions,
                                                 parent_id=individual.id)
                        survivors.append(new_ind)
                    #fully trained #todo
                    # if individual.fully_trained:
                    #     fully_trained_individuals.append(individual)
            self.generation = survivors
            print(f"cycle: {cycle}, num of survivors: {len(survivors)}")

            if len(survivors) == 0:
                break
            cycle += 1
        print("training finished")
        if do_save:
            self.save_individuals(target_collection)

    def save_individuals(self, col):
        db = get_instance(col, 'g')
        individuals_to_save = self.generation
        models = []
        for ind in individuals_to_save:
            models.append(
                GeneticIndividualModel(
                    ind.id,
                    ind.known_actions,
                    "naive0402",
                    ind.parent_id
                )
            )
        db.insert_many(models)

def models_to_individuals(models: list[GeneticIndividualModel]):
    individuals = []
    for model in models:
        individuals.append(Individual(
            model.ind_id.split(',')[0],
            model.ind_id.split(',')[1],
            known_actions=ActionHolder(model.action_set),
            parent_id=model.parent_id,
            other_parent_id=model.other_parent_id
        ))
    return individuals



#TODO review bc i changed a lot
class Genetic:

    def __init__(self, number_of_individuals: int = POPULATION_SIZE):
        self.generation = [Individual(0, _) for _ in range(number_of_individuals)]
        self.newest_generation_number = 0

    def train(self, cycles: int):
        print("starting genetic")
        #stable population:
        #each ind. takes a step, fitness check to eliminate unwanted ones
        #selection to top up population?
        #mutation
        cycle = 1
        fully_trained = []
        while cycle <= cycles:
            print(f"cycle {cycle} started")
            self.newest_generation_number = cycle
            for individual in self.generation:
                individual.act()
                # if individual.fully_trained:
                #     fully_trained.append(individual)

            self.selection()
            print(f"{len(self.generation)} individuals selected")
            self.reproduction()
            self.mutation()
        #TODO stats
        print("end genetic")

    def selection(self):
        selected = []
        for individual in self.generation:
            if self.fitness(individual):
                selected.append(individual)
        self.generation = selected

    def mutation(self):
        for individual in self.generation:
            if individual.generation == self.newest_generation_number and random() < MUTATION_RATE:
                #forget a move
                individual.known_actions.remove_action(choice(individual.known_actions.actions))

    def reproduction(self):
        # can 'newborns' be parents?
        # I say yes, bc we get more varied children that way
        while len(self.generation) < POPULATION_SIZE:
            #pick parents
            parent1 = choice(self.generation)
            parent2 = choice(self.generation)
            #create new moving function set
            inherited_actions: ActionHolder = ActionHolder()
            if random() < CROSSOVER_RATE:
                parent1, parent2 = parent2, parent1
            for action in parent1.known_actions.actions:
                inherited_actions.add_action(action)
            for action in parent2.known_actions.actions:
                inherited_actions.add_action(action)
            #create child
            new_individual = Individual(self.newest_generation_number, inherited_actions)
            #add child to generation(population)
            self.generation.append(new_individual)

    def fitness(self, individual: Individual) -> bool:
        #should we check learnt moves or just based on steps? OR just if its alive?
        #maybe other factors kick in after a certain number of gens
        #top 50%?
        if not individual.scenario.is_alive:
            return False
        #TODO otherwise, check other parameters
        return True

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
            if act.env == env and ContextHolder.contexts_eq(act.context.get_context(), context.get_context()): #todo test
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