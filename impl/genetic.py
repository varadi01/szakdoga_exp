# fitness function, the number of steps the individual managed to take
# reproduction? crossover x% of the time
# mutation? make them forget steps? change step randomly, crossover x% of the time
# selection? based on fitness function, keep individual if their fitness is over a certain percentile of all individuals(top x%)
import random

from impl.scenario import Environment, Step, ResultOfStep, SimpleGame
from random import choice
import numpy as np

POPULATION_SIZE = 20000

CROSSOVER_RATE = 0.7
MUTATION_RATE = 0.05
GENERATION_LIFETIME = 3

#TODO were making too many steps?

# TODO two approaches I reckon
# First: offspring inherits all moves, forget some, missing spots are filled with completely new individuals
# Second: surviving individuals are shuffled to create offspring, crossover, missing spots are filled via randomly selecting moves from parent generation?

class Individual:

    def __init__(self, generation: int, moving_functions = [], environment: SimpleGame = SimpleGame):
        self.generation = generation
        self.moving_functions = moving_functions
        self.fully_trained = False if len(moving_functions) < 80 else True #dunno if we need this
        self.env = environment.__init__()
        self.steps_taken = 0

    def act(self):
        action = None
        current_environment = self.env
        for env, move in self.moving_functions:
            if env == current_environment:
                action = move

        if action is None:
            action = choice((Step.UP, Step.RIGHT, Step.DOWN, Step.LEFT))
            self.moving_functions.append((current_environment, action))
            if len(self.moving_functions) > 80: #idk
                self.fully_trained = True

        self.steps_taken += 1
        self.env.make_step(action)

class IndividualNaive:

    def __init__(self, generation: int, moving_functions = [],  environment: SimpleGame = SimpleGame):
        self.generation = generation
        self.moving_functions = moving_functions
        self.fully_trained = False if len(moving_functions) < 80 else True
        self.env = environment.__init__()
        self.steps_made = 0

    def act(self):
        action = None
        current_environment = self.env
        for env, move in self.moving_functions:
            if env == current_environment:
                action = move

        if action is None:
            action = choice((Step.UP, Step.RIGHT, Step.DOWN, Step.LEFT))
            self.moving_functions.append((current_environment, action))
            if len(self.moving_functions) > 80: #idk
                self.fully_trained = True
        self.steps_made += 1
        self.env.make_step(action)


#TODO scenario PER individual, rn new individuals are playing already live scenarios, they die immediately?

class GeneticNaive:
    #TODO check if child should inherit food amount
    def __init__(self, number_of_individuals: int = POPULATION_SIZE):
        self.generation = [IndividualNaive(0) for _ in range(number_of_individuals)]

    def train(self, cycles: int):
        fully_trained_individuals = []
        cycle = 1 #for logging
        survivors = []
        while cycle <= cycles:
            print(f"cycle {cycle} started")
            # advance every individual
            for individual in [_ for _ in self.generation]: #??, necessary? to make copy i guess
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

    def train(self, cycles: int):
        #TODO REDO
        print("starting genetic")
        #stable population:
        #each ind. takes a step, fitness check to eliminate unwanted ones
        #selection to top up population?
        #mutation
        cycle = 1
        fully_trained = []
        while cycle <= cycles:
            print(f"cycle {cycle} started")

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
        #forget move? change action?
        #going to go with forget
        for individual in self.generation:
            if random.random() < MUTATION_RATE:
                #forget a move
                individual.moving_functions.remove(choice(individual.moving_functions))

    def reproduction(self):
        # can 'newborns' be parents?
        # I say yes, bc we get more varied children that way
        while len(self.generation) < POPULATION_SIZE:
            #pick parents
            #create new moving function set
            #create child
            #add child to generation(population)
            parent1 = choice(self.generation)
            parent2 = choice(self.generation)
            #really starting to wonder whether we need the eq
            #TODO make moves a separate object
            for move in parent1.moving_functions:
                other_move = parent2.moving_functions.index() if move in parent2.moving_functions else None



        # while len(new_generation) < POPULATION:
        #     parent1 = choice(new_generation)
        #     parent2 = choice(new_generation)
        #

        #inherit all, forget some approach
        new_generation = self.generation
        if IS_NAIVE:
            while POPULATION_SIZE > len(new_generation) > 0:
                i: Individual = choice(new_generation)
                #TODO forget
                i.steps_taken = 0
                i.food = 5 #TODO should inherit food amount?
                i.alive = True
                new_generation.append(i)

        # mutation?
        self.generation = new_generation

    def fitness(self, individual: Individual) -> bool:
        #should we check learnt moves or just based on steps? OR just if its alive?
        if not individual.env.is_alive:
            return False
        #TODO otherwise, check other parameters
        return True