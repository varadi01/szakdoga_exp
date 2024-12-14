# fitness function, the number of steps the individual managed to take
# reproduction? crossover x% of the time
# mutation? make them forget steps? change step randomly, crossover x% of the time
# selection? based on fitness function, keep individual if their fitness is over a certain percentile of all individuals(top x%)
from impl.scenario import Environment, Step, ResultOfStep, Game
from random import choice, random

NUMBER_OF_GENERATIONS = 30
POPULATION = 20000
CROSSOVER_RATE = 0.7
MUTATION_RATE = 0.05
GENERATION_LIFETIME = 3

IS_NAIVE = True

#TODO were making too many steps?

# TODO two approaches I reckon
# First: offspring inherits all moves, forget some, missing spots are filled with completely new individuals
# Second: surviving individuals are shuffled to create offspring, crossover, missing spots are filled via randomly selecting moves from parent generation?

class Individual:

    def __init__(self, generation: int):
        self.generation = generation
        self.moving_functions = []
        self.fully_trained = False
        self.alive = True
        self.steps_taken = 0

    def act(self, environment: Environment) -> Step:
        action = None
        for env, move in self.moving_functions:
            if env == environment:
                action = move

        if action is None:
            action = choice((Step.UP, Step.RIGHT, Step.DOWN, Step.LEFT))
            self.moving_functions.append((environment, action))

        self.steps_taken += 1
        return action


def init_generation():
    return [Individual(g) for g in range(POPULATION)]


class Genetic:

    def __init__(self, ):
        self.generation = init_generation()
        self.scenario = Game()

    def run(self):
        print("starting genetic")
        fully_trained = []

        for gen in range(NUMBER_OF_GENERATIONS):
            max_steps_taken = 0
            max_steps_taken_index = 0
            #run generation through
            #scenarios = [self.scenario() for _ in range(POPULATION)]
            scenarios = [Game() for _ in range(POPULATION)] #idk
            for _p in range(GENERATION_LIFETIME):
                #make step for every individual
                # the nth individual is playing the nth scenario
                
                for i in range(len(self.generation)):
                    individual: Individual = self.generation[i]
                    sce: Game = scenarios[i]
                    if not individual.alive:
                        continue

                    env = sce.get_environment()
                    move = individual.act(env)
                    result = sce.make_step(move)
                    if result in (ResultOfStep.STARVED, ResultOfStep.ENCOUNTERED_LION):
                        individual.alive = False
                        continue

                    #check record
                    if individual.steps_taken > max_steps_taken and individual.alive:
                        max_steps_taken = individual.steps_taken
                        max_steps_taken_index = i

                    # if fully trained
                    if len(individual.moving_functions) >= 81:
                        s = individual
                        fully_trained.append(s) #TODO fully trained, place elsewhere
                #end for

            survived_num = len([_ for _ in self.generation if _.alive])
            if survived_num > 0:
                print(f"{survived_num} individuals survived this ({gen+1}.) generation cycle")
                print(f"maximum steps reached were {max_steps_taken} with  {len(self.generation[max_steps_taken_index].moving_functions)} moves learned")
            else:
                print("no survivors remain")
                break

            #selection
            new_generation = self.selection()

            #reproduction
            new_generation = self.reproduction(new_generation)

            #mutation?

            self.generation = new_generation

            #end for

        #end for
        #TODO save fully trained
        print(f"{len(fully_trained)} individuals were fully trained")


    def selection(self):
        new_generation = []
        # TODO fitness check
        if IS_NAIVE:
            for i in range(len(self.generation)):
                individual = self.generation[i]
                if individual.alive:
                    new_generation.append(individual)
        else:
            pass
        return new_generation

    def mutation(self):
        pass

    def reproduction(self, new_generation):
        # while len(new_generation) < POPULATION:
        #     parent1 = choice(new_generation)
        #     parent2 = choice(new_generation)
        #

        #inherit all, forget some approach
        if IS_NAIVE:
            while POPULATION > len(new_generation) > 0: #TODO training should stop if all individuals have died
                i = choice(new_generation)
                #TODO forget
                new_generation.append(i)

        # mutation?
        return new_generation

    def fitness(self, individual: Individual):
        #should we check learnt moves or just based on steps? OR just if its alive?
        return 1 if individual.alive else 0