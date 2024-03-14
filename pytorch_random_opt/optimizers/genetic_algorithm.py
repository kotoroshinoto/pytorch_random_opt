from typing import Iterable, List, Tuple, Dict, Any

import numpy as np
import torch
from torch.optim import Optimizer


class GeneticAlgorithm(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.Tensor],
            lr=0.0001, # noqa
            pop_size=200,
            pop_breed_percent=0.75,
            elite_dreg_ratio=0.99,
            minimum_elites=0,
            minimum_dregs=0,
            mutation_prob=0.1,
            max_attempts=10,
            max_iters=float('inf'),
            random_state=None,
            hamming_factor=0.0,
            hamming_decay_factor=None
    ):
        self.np_generator = np.random.RandomState()
        self.torch_generator = torch.Generator()
        if random_state is not None:
            self.np_generator.seed(random_state)
            self.torch_generator.manual_seed(random_state)

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

        self.params_flat, self.params_flat_shapes, self.params_other_kv = self.__class__._flatten_params(
            self.param_groups
        )
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
    def _flatten_params(
            param_groups: List[Dict[str, Any]]
    ) -> Tuple[torch.Tensor, List[List[torch.Size]], List[Dict[str, Any]]]:
        things_to_torch_cat: List[torch.Tensor] = []
        shapes: List[List[torch.Size]] = []
        other_keys_and_values: List[Dict[str, Any]] = []
        for i, param_group in enumerate(param_groups):
            other_kv = dict()
            shapes_i = []
            # print(f"Parameter Group {i}:")
            # print("Parameters:")
            for pg_params in param_group['params']:
                for p in pg_params:
                    things_to_torch_cat.append(p.view(-1))
                    shapes_i.append(p.shape)
                # print(pg_params)
            # print("Options:")
            for key, value in param_group.items():
                if key != 'params':
                    other_kv[key] = value
            #         print(f"{key}: {value}")
            other_keys_and_values.append(other_kv)
            shapes.append(shapes_i)
        # raise ValueError("STOP")
        # return torch.cat([p.view(-1) for p in params_iter])
        if len(shapes) != len(other_keys_and_values):
            raise ValueError("Shapes list and other_kv list are not the same size")
        return torch.cat(things_to_torch_cat), shapes, other_keys_and_values

    @staticmethod
    def _unflatten_params(
            flat_params: torch.Tensor,
            shapes: List[List[torch.Size]],
            other_kv: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        unflattened_params: List[Dict[str, Any]] = []
        flat_param_idx = 0
        if len(shapes) != len(other_kv):
            raise ValueError("Shapes list and other_kv list are not the same size")
        for i in range(len(shapes)):
            group_params: List[torch.Tensor] = []
            group_shapes = shapes[i]
            group_kv = other_kv[i]
            for shape in group_shapes:
                numel = int(torch.tensor(shape).prod().item())  # Calculate number of elements in tensor
                param_data = flat_params[flat_param_idx:flat_param_idx + numel].view(shape)  # Extract data
                flat_param_idx += numel  # Move to next chunk of flattened parameters
                group_params.append(param_data)  # Append unflattened parameter
            param_dict: Dict[str, Any] = {'params': group_params}
            param_dict.update(group_kv)
            unflattened_params.append(param_dict)
        return unflattened_params

    def _update_params(self, state: torch.Tensor):
        unflat_state = self.__class__._unflatten_params(state, self.params_flat_shapes, self.params_other_kv)
        self.param_groups = unflat_state

    def step(self, closure=None):
        if closure is None:
            raise ValueError("This algorithm requires a loss function")
        fitnesses = torch.tensor(self._execute_closure_multi(self.population, closure))

        # Calculate breeding probabilities
        mating_probabilities = self._calculate_breeding_probabilities(fitnesses=fitnesses)

        # Create next generation
        next_generation, best_state, best_fitness = self._create_next_generation(
            mating_probabilities,
            fitnesses=fitnesses
        )

        # Update population
        self.population = next_generation

        # Decay hamming factor if applicable
        self._hamming_decay()

        self._update_params(best_state)

    def _genetic_alg_select_parents(self, population: List[torch.Tensor], mating_probabilities):
        if self.hamming_factor > 0.01:
            selected = torch.multinomial(mating_probabilities, 1, replacement=True)
            p1 = population[selected[0]]
            hamming_distances = torch.tensor([torch.abs(p1 - p2).sum() / len(p1) for p2 in population])
            hfa = self.hamming_factor / (1.0 - self.hamming_factor)

            hamming_factor_adjusted_probs = (hamming_distances * hfa) * mating_probabilities
            hamming_factor_adjusted_probs /= hamming_distances.sum()

            selected2 = torch.multinomial(hamming_factor_adjusted_probs, 1, replacement=True)
            p2 = population[selected2[0]]
        else:
            selected = torch.multinomial(mating_probabilities, 2, replacement=True)
            p1 = population[selected[0]]
            p2 = population[selected[1]]
        return p1, p2

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
        random_state = torch.randn(self.params_flat.shape, generator=self.torch_generator)
        return random_state

    def _calculate_breeding_probabilities(self, fitnesses):
        mating_probabilities = fitnesses / fitnesses.sum()
        return mating_probabilities

    def _execute_closure(self, state: torch.Tensor, closure):
        original_param_group = self.param_groups
        self._update_params(state)
        result = closure()
        self.param_groups = original_param_group
        return result

    def _execute_closure_multi(self, states: Iterable[torch.Tensor], closure):
        results = []
        original_param_group = self.param_groups
        for state in states:
            self._update_params(state)
            result = closure()
            results.append(result)
        self.param_groups = original_param_group
        return results

    def _create_next_generation(self, mating_probabilities, fitnesses) -> Tuple[List[torch.Tensor], torch.Tensor, Any]:
        # Create next generation of population
        next_gen = []

        # Select breeding parents and perform crossover
        breeding_pop_size = int(self.pop_size * self.pop_breed_percent) - (self.minimum_elites + self.minimum_dregs)
        elite_size = max(int(breeding_pop_size * self.elite_dreg_ratio), self.minimum_elites)
        dreg_size = max(breeding_pop_size - elite_size, self.minimum_dregs)

        for _ in range(breeding_pop_size):
            parent_1, parent_2 = self._genetic_alg_select_parents(self.population, mating_probabilities)
            child = self._crossover(parent_1, parent_2)
            child = self._mutate(child)
            next_gen.append(child)

        sorted_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])
        sorted_fitnesses = [fitnesses[i] for i in sorted_indices]
        sorted_population = [self.population[i] for i in sorted_indices]

        elites = sorted_population[:elite_size]
        dregs = sorted_population[-dreg_size:]

        # Fill remaining population with elites and dregs
        next_gen.extend(elites)
        next_gen.extend(dregs)

        best_state = sorted_population[0]
        best_fitness = sorted_fitnesses[0]

        return next_gen, best_state, best_fitness

    def _crossover(self, parent_1: torch.Tensor, parent_2: torch.Tensor):
        if len(parent_1) > 1:
            # Randomly select a crossover point
            crossover_point = self.np_generator.randint(len(parent_1) - 1)
            # Create the child by splicing the parents' chromosomes at the crossover point
            child = torch.cat((parent_1[:crossover_point], parent_2[crossover_point:]))
        elif self.np_generator.randint(2) == 0:

            child = torch.clone(parent_1)
        else:
            child = torch.clone(parent_2)
        return child

    def _mutate(self, child: torch.Tensor) -> torch.Tensor:
        mutated_child = child.clone()
        for idx, val in enumerate(mutated_child.view(-1)):
            if self.np_generator.rand() < self.mutation_prob:
                mutated_child.view(-1)[idx] = self.np_generator.uniform(-1, 1)
        return mutated_child
