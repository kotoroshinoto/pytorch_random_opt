from typing import Iterable, List, Tuple

import numpy as np
import torch
from torch.optim import Optimizer


class GeneticAlgorithm(Optimizer):
    def __init__(self, params: Iterable[torch.Tensor], pop_size=200, pop_breed_percent=0.75, elite_dreg_ratio=0.99,
                 minimum_elites=0, minimum_dregs=0, mutation_prob=0.1,
                 max_attempts=10, max_iters=float('inf'), random_state=None,
                 hamming_factor=0.0, hamming_decay_factor=None):

        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)

        if pop_size < 0:
            raise ValueError("pop_size must be a positive integer.")

        # Define defaults dictionary
        defaults = dict(pop_size=pop_size, pop_breed_percent=pop_breed_percent,
                        elite_dreg_ratio=elite_dreg_ratio, minimum_elites=minimum_elites,
                        minimum_dregs=minimum_dregs, mutation_prob=mutation_prob,
                        max_attempts=max_attempts, max_iters=max_iters,
                        random_state=random_state, hamming_factor=hamming_factor,
                        hamming_decay_factor=hamming_decay_factor)

        # Initialize the optimizer with the model parameters and defaults
        super(GeneticAlgorithm, self).__init__(params, defaults)

        self.params = params
        self.params_flat = self.__class__._flatten_params(self.params)
        self.pop_size = pop_size
        self.pop_breed_percent = pop_breed_percent
        self.elite_dreg_ratio = elite_dreg_ratio
        self.minimum_elites = minimum_elites
        self.minimum_dregs = minimum_dregs
        self.mutation_prob = mutation_prob
        self.max_attempts = max_attempts
        self.max_iters = max_iters
        self.random_state = random_state
        self.hamming_factor = hamming_factor
        self.hamming_decay_factor = hamming_decay_factor

        # Initialize population
        self.population: List[torch.Tensor] = self._initialize_population()

    @staticmethod
    def _flatten_params(params_iter: Iterable[torch.Tensor]) -> torch.Tensor:
        return torch.cat([p.view(-1) for p in params_iter])

    @staticmethod
    def _unflatten_params(flat_params: torch.Tensor, shapes: List[torch.Size]) -> List[torch.Tensor]:
        unflattened_params = []
        start_index = 0
        for shape in shapes:
            numel = shape.numel()
            unflattened_params.append(flat_params[start_index:start_index + numel].view(shape))
            start_index += numel
        return unflattened_params

    def step(self, closure=None):
        if closure is None:
            raise ValueError("This algorithm requires a loss function")
        # Calculate breeding probabilities
        mating_probabilities = self._calculate_breeding_probabilities(closure=closure)

        # Create next generation
        next_generation = self._create_next_generation(mating_probabilities, closure=closure)

        # Update population
        self.population = next_generation

        # Decay hamming factor if applicable
        self._hamming_decay()

        best_state, best_fitness = self._get_best_state_and_fitness(closure=closure)

        return best_state, best_fitness

    def _genetic_alg_select_parents(self, population: List[torch.Tensor], mating_probabilities):
        selected = torch.multinomial(mating_probabilities, 2, replacement=True)
        p1 = population[selected[0]]
        p2 = population[selected[1]]
        return p1, p2

    def _get_hamming_distance_default(self, population: List[torch.Tensor], p1):
        hamming_distances = torch.tensor([torch.count_nonzero(p1 != p2).item() / len(p1) for p2 in population])
        return hamming_distances

    def _get_hamming_distance_float(self, population: List[torch.Tensor], p1):
        hamming_distances = torch.tensor([torch.abs(p1 - p2).sum().item() / len(p1) for p2 in population])
        return hamming_distances

    def _hamming_decay(self):
        if self.hamming_decay_factor is not None and self.hamming_factor > 0.0:
            self.hamming_factor *= self.hamming_decay_factor
            self.hamming_factor = max(min(self.hamming_factor, 1.0), 0.0)

    def _initialize_population(self) -> List[torch.Tensor]:
        population = []
        for _ in range(self.pop_size):
            state = self._generate_random_state()
            population.append(state)
        return population

    def _generate_random_state(self) -> torch.Tensor:
        random_state = torch.rand_like(self.params_flat)
        return random_state

    def _calculate_breeding_probabilities(self, closure):
        fitnesses = torch.tensor([closure(state) for state in self.population])
        mating_probabilities = fitnesses / fitnesses.sum()
        return mating_probabilities

    def _create_next_generation(self, mating_probabilities, closure) -> List[torch.Tensor]:
        # Create next generation of population
        next_gen = []

        # Select breeding parents and perform crossover
        breeding_pop_size = int(self.pop_size * self.pop_breed_percent) - (self.minimum_elites + self.minimum_dregs)
        elite_size = max(int(breeding_pop_size * self.elite_dreg_ratio), self.minimum_elites)
        dreg_size = max(breeding_pop_size - elite_size, self.minimum_dregs)

        for _ in range(breeding_pop_size):
            parent_1, parent_2 = self._genetic_alg_select_parents(self.population, mating_probabilities)
            child = self._crossover(parent_1, parent_2)
            next_gen.append(child)

        # Fill remaining population with elites and dregs
        next_gen.extend(self._select_elites(elite_size, closure=closure))
        next_gen.extend(self._select_dregs(dreg_size, closure=closure))

        return next_gen

    def _crossover(self, parent_1: torch.Tensor, parent_2: torch.Tensor):
        if len(parent_1) > 1:
            # Randomly select a crossover point
            crossover_point = np.random.randint(len(parent_1) - 1)
            # Create the child by splicing the parents' chromosomes at the crossover point
            child = torch.cat((parent_1[:crossover_point], parent_2[crossover_point:]))
        elif np.random.randint(2) == 0:

            child = torch.clone(parent_1)
        else:
            child = torch.clone(parent_2)
        return child

    def _mutate(self, child: torch.Tensor) -> torch.Tensor:
        mutated_child = child.clone()
        for idx, val in enumerate(mutated_child.view(-1)):
            if np.random.rand() < self.mutation_prob:
                mutated_child.view(-1)[idx] = np.random.uniform(-1, 1)
        return mutated_child

    def _select_elites(self, elite_size, closure) -> List[torch.Tensor]:
        sorted_population = sorted(self.population, key=lambda state: -closure(state))
        elites = sorted_population[:elite_size]
        return elites

    def _select_dregs(self, dreg_size, closure) -> List[torch.Tensor]:
        sorted_population = sorted(self.population, key=lambda state: closure(state))
        dregs = sorted_population[:dreg_size]
        return dregs

    def _get_best_state_and_fitness(self, closure) -> Tuple[torch.Tensor, torch.Tensor]:
        best_state, best_fitness = max(self.population, key=lambda state: closure(state))
        return best_state, best_fitness
