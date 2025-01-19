# fitness function, the number of steps the individual managed to take
# reproduction? crossover x% of the time
# mutation? make them forget steps? change step randomly, crossover x% of the time
# selection? based on fitness function, keep individual if their fitness is over a certain percentile of all individuals(top x%)

from impl.scenario import Environment, Step, SimpleGame, ContextHolder, ContextBasedGame, PieceOfContext
from random import choice, random

POPULATION_SIZE = 20000

CROSSOVER_RATE = 0.7
MUTATION_RATE = 0.05

#TODO meld together individual

# two approaches I reckon
# First: offspring inherits all moves, forget some, missing spots are filled with completely new individuals
# Second: surviving individuals are shuffled to create offspring, crossover, missing spots are filled via randomly selecting moves from parent generation?

class Action:

    def __init__(self, env: Environment, step: Step):
        self.env = env
        self.step = step

    def env_eq(self, other):
        return self.env == other.env

    def __eq__(self, other):
        return self.env == other.env and self.step == other.step


class ActionHolder:

    def __init__(self, actions: list[Action] = None):
        if actions is None:
            actions = []
        self.actions = actions

    def add_action(self, action: Action):
        if self.is_env_known(action.env):
            return
        self.actions.append(action)

    def remove_action(self, action):
        try:
            self.actions.remove(action)
        except ValueError:
            raise Exception("tried to delete non-existent action")

    def is_env_known(self, env: Environment) -> bool:
        known = False
        for act in self.actions:
            if act.env == env:
                known = True
        return known

    def get_action_for_env(self, env: Environment) -> Action:
        action = None
        for act in self.actions:
            if act.env == env:
                action = act
        if action is not None:
            return action
        raise Exception("action not known")


class Individual:

    def __init__(self, generation: int, known_actions: ActionHolder = ActionHolder(), scenario: SimpleGame = SimpleGame):
        self.generation = generation
        self.known_actions = known_actions
        self.fully_trained = False if len(known_actions.actions) < 80 else True #dunno if we need this
        self.scenario = scenario.__init__()
        self.steps_taken = 0

    def act(self):
        #step_to_take = None  #maybe remove?
        current_environment = self.scenario.get_environment()
        if self.known_actions.is_env_known(current_environment):
            step_to_take = self.known_actions.get_action_for_env(current_environment)
        else:
            step_to_take = choice((Step.UP, Step.RIGHT, Step.DOWN, Step.LEFT))
            self.known_actions.add_action(Action(current_environment, step_to_take))
            if len(self.known_actions.actions) > 80: #idk
                self.fully_trained = True
        self.steps_taken += 1
        self.scenario.make_step(step_to_take)

class IndividualNaive:
    #todo refactor known_action?, easier to read maybe
    def __init__(self, generation: int, moving_functions = None, scenario: SimpleGame = SimpleGame):
        if moving_functions is None:
            moving_functions = []
        self.generation = generation
        self.moving_functions = moving_functions
        self.fully_trained = False if len(moving_functions) < 80 else True
        self.scenario = scenario.__init__()
        self.steps_made = 0

    def act(self):
        step_to_take = None
        current_environment = self.scenario.get_environment()
        for act in self.moving_functions:
            if act.env == current_environment:
                step_to_take = act.step

        if step_to_take is None:
            step_to_take = choice((Step.UP, Step.RIGHT, Step.DOWN, Step.LEFT))
            self.moving_functions.append(Action(current_environment, step_to_take))
            if len(self.moving_functions) > 80: #idk
                self.fully_trained = True
        self.steps_made += 1
        self.scenario.make_step(step_to_take)


class GeneticNaive:
    #TODO check if child should inherit food amount
    def __init__(self, number_of_individuals: int = POPULATION_SIZE):
        self.generation = [IndividualNaive(0) for _ in range(number_of_individuals)]

    def train(self, cycles: int):
        fully_trained_individuals = []
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
                        new_ind = IndividualNaive(cycle, individual.moving_functions)
                        survivors.append(new_ind)
                    #fully trained
                    if individual.fully_trained:
                        fully_trained_individuals.append(individual)
            self.generation = survivors
            print(f"cycle: {cycle}, num of survivors: {len(survivors)}")

            if len(survivors) == 0:
                break
            cycle += 1
        #TODO store stuff
        print("training finished")

class Genetic:

    def __init__(self, number_of_individuals: int = POPULATION_SIZE):
        self.generation = [Individual(0) for _ in range(number_of_individuals)]
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
                if individual.fully_trained:
                    fully_trained.append(individual)

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
        if not individual.scenario.is_alive:
            return False
        #TODO otherwise, check other parameters
        return True


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
        return self.step == other.step and self.env_and_context_eq(other) ##todo, maybe refactor eqs to static

class ActionWithContextHolder:
    #TODO not happy with this, refactor if possible

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
    #TODO test, refactor maybe

    def __init__(self, generation: int, known_actions: ActionWithContextHolder = ActionWithContextHolder(),  scenario: ContextBasedGame  = ContextBasedGame):
        self.generation = generation
        self.known_actions = known_actions
        self.scenario = scenario.__init__() #TODO this is not gonna be okay, if i wanna give them predefined scenarios!!!!!!
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
    #TODO check if child should inherit food amount
    def __init__(self, number_of_individuals: int = POPULATION_SIZE):
        self.generation = [IndividualWithContext(0) for _ in range(number_of_individuals)]

    def train(self, cycles: int):
        #todo, since fully training is unreasonable, what do we do?
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
        #TODO store stuff
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
        #TODO stats
        print("end genetic")

    def selection(self):
        selected = []
        for individual in self.generation:
            if self.fitness(individual):
                selected.append(individual)
        self.generation = selected

    def mutation(self):
        #todo test
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
        #TODO otherwise, check other parameters
        return True