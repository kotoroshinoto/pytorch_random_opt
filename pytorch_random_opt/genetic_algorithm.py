import torch
from torch.optim import Optimizer

class GeneticAlgorithm(Optimizer):
    def __init__(self, problem, pop_size=200, pop_breed_percent=0.75, elite_dreg_ratio=0.99,
                 minimum_elites=0, minimum_dregs=0, mutation_prob=0.1,
                 max_attempts=10, max_iters=float('inf'), random_state=None,
                 state_fitness_callback=None, callback_user_info=None,
                 hamming_factor=0.0, hamming_decay_factor=None):
        if not isinstance(problem, torch.nn.Module):
            raise TypeError("The 'problem' parameter must be a torch.nn.Module instance.")

        if pop_size < 0:
            raise ValueError("pop_size must be a positive integer.")

        # Initialize the optimizer with an empty parameter list
        super(GeneticAlgorithm, self).__init__([], {})

        self.problem = problem
        self.pop_size = pop_size
        self.pop_breed_percent = pop_breed_percent
        self.elite_dreg_ratio = elite_dreg_ratio
        self.minimum_elites = minimum_elites
        self.minimum_dregs = minimum_dregs
        self.mutation_prob = mutation_prob
        self.max_attempts = max_attempts
        self.max_iters = max_iters
        self.random_state = random_state
        self.state_fitness_callback = state_fitness_callback
        self.callback_user_info = callback_user_info
        self.hamming_factor = hamming_factor
        self.hamming_decay_factor = hamming_decay_factor

        # Initialize population
        self.population = self._initialize_population()

    def step(self, closure=None):
        # Calculate breeding probabilities
        mating_probabilities = self._calculate_breeding_probabilities()

        # Create next generation
        next_generation = self._create_next_generation(mating_probabilities)

        # Update population
        self.population = next_generation

        # Decay hamming factor if applicable
        self._hamming_decay()

        # Invoke callback
        best_state, best_fitness = self._get_best_state_and_fitness()

        return best_state, best_fitness

    def _genetic_alg_select_parents(self, pop_size, population, mating_probabilities):
        selected = torch.multinomial(mating_probabilities, 2, replacement=True)
        p1 = population[selected[0]]
        p2 = population[selected[1]]
        return p1, p2

    def _get_hamming_distance_default(self, population, p1):
        hamming_distances = torch.tensor([torch.count_nonzero(p1 != p2).item() / len(p1) for p2 in population])
        return hamming_distances

    def _get_hamming_distance_float(self, population, p1):
        hamming_distances = torch.tensor([torch.abs(p1 - p2).sum().item() / len(p1) for p2 in population])
        return hamming_distances

    def _hamming_decay(self):
        if self.hamming_decay_factor is not None and self.hamming_factor > 0.0:
            self.hamming_factor *= self.hamming_decay_factor
            self.hamming_factor = max(min(self.hamming_factor, 1.0), 0.0)

    def _initialize_population(self):
        population = []
        for _ in range(self.pop_size):
            state = torch.rand_like(self.problem.parameters())
            population.append(state)
        return population

    def _calculate_breeding_probabilities(self, population):
        fitnesses = torch.tensor([self.problem.eval_fitness(state) for state in population])
        mating_probabilities = fitnesses / fitnesses.sum()
        return mating_probabilities

    def _create_next_generation(self, population, mating_probabilities):
        # Create next generation of population
        next_gen = []

        # Select breeding parents and perform crossover
        breeding_pop_size = int(self.pop_size * self.pop_breed_percent) - (self.minimum_elites + self.minimum_dregs)
        elite_size = max(int(breeding_pop_size * self.elite_dreg_ratio), self.minimum_elites)
        dreg_size = max(breeding_pop_size - elite_size, self.minimum_dregs)

        for _ in range(breeding_pop_size):
            parent_1, parent_2 = self._genetic_alg_select_parents(self.pop_size, population, mating_probabilities)
            child = self._crossover(parent_1, parent_2)
            next_gen.append(child)

        # Fill remaining population with elites and dregs
        next_gen.extend(self._select_elites(population, elite_size))
        next_gen.extend(self._select_dregs(population, dreg_size))

        return next_gen

    def _crossover(self, parent_1, parent_2):
        mask = torch.rand_like(parent_1) < 0.5
        child = torch.where(mask, parent_1, parent_2)
        return child

    def _select_elites(self, population, elite_size):
        sorted_population = sorted(population, key=lambda state: -self.problem.eval_fitness(state))
        elites = sorted_population[:elite_size]
        return elites

    def _select_dregs(self, population, dreg_size):
        sorted_population = sorted(population, key=lambda state: self.problem.eval_fitness(state))
        dregs = sorted_population[:dreg_size]
        return dregs

    def _invoke_callback(self, iteration, attempt, done, state, fitness, fitness_evaluations, curve):
        if self.state_fitness_callback is not None:
            max_attempts_reached = (attempt == self.max_attempts) or (iteration == self.max_iters) or self.problem.can_stop()
            continue_iterating = self.state_fitness_callback(iteration=iteration,
                                                              attempt=attempt,
                                                              done=max_attempts_reached,
                                                              state=state,
                                                              fitness=fitness,
                                                              fitness_evaluations=fitness_evaluations,
                                                              curve=curve,
                                                              user_data=self.callback_user_info)
            return continue_iterating

    def _get_best_state_and_fitness(self, population):
        best_state, best_fitness = max(population, key=lambda state: self.problem.eval_fitness(state))
        return best_state, best_fitness
