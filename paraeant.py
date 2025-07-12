import random
import copy
import numpy as np
import math
from typing import List, Tuple, Dict, Optional, Set, Union
from enum import Enum
from genome import Genome
from neural_network_evaluator import NeuralNetworkEvaluator
from genome import VertexGene, InputGene, ForwardJumperGene, RecurrentJumperGene, Gene

class MutationType(Enum):
    ADD_SUBNETWORK = 0
    ADD_CONNECTION = 1
    REMOVE_CONNECTION = 2
    
class EANT:
    def __init__(self, population_size: int, input_count: int, output_count: int):
        self.population_size = population_size
        self.population = []
        self.tree_boundaries_list = []
        self.best_genome = None
        self.best_tree_boundaries = None
        self.best_fitness = float('-inf')
        self.input_count = input_count
        self.output_count = output_count
        self.evaluations = 0
        self.mutation_prob = 0.4
        self.structural_mutation_prob = 0.4
        self.structural_mutation_prob_start = 0.2

        self.buffer_length = 5        # How many generations to look back
        self.improvement_threshold = 0.2  # Minimum improvement required
        self.fitness_history = [] 
        
        # Parameters for self-adaptive mutation
        self.min_sigma = 0.01  # ε₀ value from equation 2.6
        self.initial_sigma = 0.5  # Initial step size

    def initialize_minimal_population(self) -> None:
        """Initialize population with minimal networks"""
        self.population = []
        self.tree_boundaries_list = []
        
        for _ in range(self.population_size):
            genome = Genome(depth=1)
            tree_boundaries = genome.build_expression_tree(self.input_count, self.output_count)
            self.population.append(genome)
            self.tree_boundaries_list.append(tree_boundaries)
        
        print(f"Generated {len(self.population)} minimal networks")
        self.initial_neuron_count = sum(1 for genome in self.population 
                                     for gene in genome.genes if isinstance(gene, VertexGene))
        self.fitness_history = []
        self.protected_genomes = []
        self.protected_tree_boundaries = []
        self.protection_timers = []

    def evaluate_population(self, X: List[List[float]], y: List[float]) -> List[float]:
        """Evaluate all genomes in the population"""
        mse_values = []
        for genome, tree_boundaries in zip(self.population, self.tree_boundaries_list):
            evaluator = NeuralNetworkEvaluator(genome, tree_boundaries)
            total_error = 0.0
            for j in range(len(X)):
                inputs = {f"i{k}": X[j][k] for k in range(len(X[j]))}
                outputs = evaluator.evaluate(inputs)
                error = (outputs[0] - y[j])**2
                total_error += error
                evaluator.reset_state()
            
            mse = total_error / len(X)
            mse_values.append(mse)
            
            fitness = -mse
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_genome = genome.copy()
                self.best_tree_boundaries = tree_boundaries.copy()
            
            self.evaluations += 1
        return mse_values

    def parametric_mutation(self, population: List[Genome], fitness_values: Dict[Genome, float]) -> List[Genome]:
        """Apply self-adaptive parametric mutation to population based on cluster fitness"""
        clusters = {}
        for genome in population:
            signature = self._get_structure_signature(genome)
            if signature not in clusters:
                clusters[signature] = []
            clusters[signature].append(genome)
        
        # Calculate average fitness per cluster
        cluster_fitness = {}
        total_fitness_sum = 0
        
        for signature, genomes in clusters.items():
            # Get fitness values for genomes in this cluster
            cluster_genome_fitness = [fitness_values.get(g, float('-inf')) for g in genomes]
            
            # Calculate average fitness (Fk)
            avg_fitness = sum(cluster_genome_fitness) / len(cluster_genome_fitness)
            cluster_fitness[signature] = avg_fitness
            total_fitness_sum += avg_fitness  # Total fitness sum (Ft)
        
        # Create new population through fitness-proportional mutation
        new_population = []
        
        for signature, genomes in clusters.items():
            # Calculate proportion of resources for this cluster: (Fk/Ft)
            if total_fitness_sum == 0:  # Avoid division by zero
                proportion = 1.0 / len(clusters)
            else:
                proportion = cluster_fitness[signature] / total_fitness_sum
            
            # Calculate number of evaluations: ne = (Fk/Ft)PS
            num_evaluations = max(1, int(proportion * self.population_size))
            
            # Select best genome from cluster as representative
            best_genome = max(genomes, key=lambda g: fitness_values.get(g, float('-inf')))
            
            # Apply self-adaptive mutation multiple times based on allocated resources
            for _ in range(num_evaluations):
                # Create a copy for mutation
                mutated_genome = best_genome.copy()
                
                # Initialize sigmas if they don't exist or have wrong length
                if not hasattr(mutated_genome, 'sigmas') or len(mutated_genome.sigmas) != len(mutated_genome.genes):
                    mutated_genome.initialize_sigmas(self.initial_sigma)
                
                # Apply self-adaptive mutation
                self._apply_self_adaptive_mutation(mutated_genome)
                
                new_population.append(mutated_genome)
        
        # Ensure population size is maintained
        if len(new_population) < self.population_size:
            # If we have too few genomes, clone the best ones
            sorted_genomes = sorted(population, 
                                key=lambda g: fitness_values.get(g, float('-inf')), 
                                reverse=True)
            while len(new_population) < self.population_size:
                for g in sorted_genomes:
                    if len(new_population) < self.population_size:
                        mutated = g.copy()
                        self._apply_self_adaptive_mutation(mutated)
                        new_population.append(mutated)
                    else:
                        break
        elif len(new_population) > self.population_size:
            # If we have too many, keep the best ones
            new_population = new_population[:self.population_size]
        
        return new_population
    
    def _apply_self_adaptive_mutation(self, genome):
        """Apply the self-adaptive mutation mechanism based on equations 2.4-2.6"""
        n = len(genome.genes)
        
        # Calculate tau values as per the equations
        tau_prime = 1.0 / math.sqrt(2.0 * n)
        tau = 1.0 / math.sqrt(2.0 * math.sqrt(n))
        
        # Global random factor that affects all step sizes
        global_factor = np.random.normal(0, 1)
        
        # Update step sizes (sigmas) for each gene
        for i in range(len(genome.genes)):
            # Generate individual random factor
            individual_factor = np.random.normal(0, 1)
            
            # Update sigma according to equation 2.4
            genome.sigmas[i] *= math.exp(tau_prime * global_factor + tau * individual_factor)
            
            # Apply boundary rule (equation 2.6)
            if genome.sigmas[i] < self.min_sigma:
                genome.sigmas[i] = self.min_sigma
            
            # Apply mutation to weights according to equation 2.5
            if random.random() < self.mutation_prob:
                current_weight = genome.genes[i].get_weight()
                random_factor = np.random.normal(0, 1)
                new_weight = current_weight + genome.sigmas[i] * random_factor
                genome.genes[i].set_weight(new_weight)
        
        return genome
    
    def structural_mutation(self, genome: Genome, mutation_prob=None) -> Genome:
        """Add/remove structure through mutation"""
        if mutation_prob is None:
            mutation_prob = self.structural_mutation_prob

        # Adjust probability based on network size
        current_neuron_count = sum(1 for gene in genome.genes if isinstance(gene, VertexGene))
        if current_neuron_count > 0 and self.initial_neuron_count > 0:
            adjusted_prob = self.structural_mutation_prob_start * (self.initial_neuron_count / current_neuron_count)
            mutation_prob = min(adjusted_prob, 1.0)
            
        new_genome = genome.copy()
        vertex_genes = [gene for gene in new_genome.genes if isinstance(gene, VertexGene)]
        if not vertex_genes:
            return new_genome
        vertex_to_mutate = random.choice(vertex_genes)
        rand = random.random()
        if rand < 0.33:  
            mutation_type = MutationType.ADD_CONNECTION
        elif rand < 0.67:  
            mutation_type = MutationType.ADD_SUBNETWORK
        else:  
            mutation_type = MutationType.REMOVE_CONNECTION
        if mutation_type == MutationType.ADD_SUBNETWORK:
            self._add_sub_network(new_genome, vertex_to_mutate)
            print(f"Added sub-network to vertex {vertex_to_mutate.id}")
        elif mutation_type == MutationType.ADD_CONNECTION:
            self._add_jumper_connection(new_genome, vertex_to_mutate)
            print(f"Added connection to vertex {vertex_to_mutate.id}")
        else:
            if vertex_to_mutate.arity > 1:
                self._remove_jumper_connection(new_genome, vertex_to_mutate)
                print(f"Removed connection from vertex {vertex_to_mutate.id}")
            else:
                self._add_jumper_connection(new_genome, vertex_to_mutate)
                print(f"Can't remove from vertex {vertex_to_mutate.id} with arity 1, added connection instead")
        valid, error = new_genome.is_valid()
        if valid:
            # Update sigmas to match new genome structure
            if len(new_genome.sigmas) != len(new_genome.genes):
                # Add new sigmas for any new genes
                old_len = len(new_genome.sigmas)
                new_len = len(new_genome.genes)
                if old_len < new_len:
                    new_genome.sigmas.extend([self.initial_sigma] * (new_len - old_len))
                else:
                    new_genome.sigmas = new_genome.sigmas[:new_len]
            return new_genome
        else:
            print(f"Mutation rejected - invalid genome: {error}")
            return genome
        
    def _add_jumper_connection(self, genome: Genome, target_vertex: VertexGene) -> None:
        """Add a forward jumper connection to a vertex (no recurrent jumpers)"""
        target_idx = None
        for i, gene in enumerate(genome.genes):
            if isinstance(gene, VertexGene) and gene.id == target_vertex.id:
                target_idx = i
                break
        
        if target_idx is None:
            return
        s_values = genome.calculate_s_values()
        valid_source_vertices = []
        for i, gene in enumerate(genome.genes):
            if isinstance(gene, VertexGene) and gene.id != target_vertex.id:
                if i < target_idx and s_values[i] < s_values[target_idx]:
                    valid_source_vertices.append(gene)
        if valid_source_vertices:
            source_vertex = random.choice(valid_source_vertices)
            jumper_gene = ForwardJumperGene(source=source_vertex.id)
            jumper_gene.set_weight(random.uniform(-1.0, 1.0) * 2)
            genome.genes.insert(target_idx + 1, jumper_gene)
            target_vertex.arity += 1
        else:
            self._add_input_connection(genome, target_vertex)
        
    def _add_input_connection(self, genome: Genome, target_vertex: VertexGene) -> None:
        """Add an input connection to a vertex when no other vertices available"""
        target_idx = None
        for i, gene in enumerate(genome.genes):
            if isinstance(gene, VertexGene) and gene.id == target_vertex.id:
                target_idx = i
                break
        
        if target_idx is None:
            return
        input_idx = random.randrange(self.input_count)
        input_gene = InputGene(label=f"i{input_idx}")
        input_gene.set_weight(random.uniform(-1.0, 1.0) * 2)  # Strong initial weight
        genome.genes.insert(target_idx + 1, input_gene)
        target_vertex.arity += 1
    
    def _add_sub_network(self, genome: Genome, parent_vertex: VertexGene) -> None:
        """Add a sub-network (hidden neuron with connections) to a vertex"""
        parent_idx = None
        for i, gene in enumerate(genome.genes):
            if isinstance(gene, VertexGene) and gene.id == parent_vertex.id:
                parent_idx = i
                break
        if parent_idx is None:
            return
        new_vertex = VertexGene(gene_id=genome.next_id, arity=0)
        num_inputs = random.randint(2, 3)
        new_vertex.arity = num_inputs
        activation_options = ['relu', 'sigmoid', 'tanh', 'leaky_relu']
        new_vertex.activation = random.choice(activation_options)
        new_genes = [new_vertex]
        for _ in range(num_inputs):
            input_idx = random.randrange(self.input_count)
            input_gene = InputGene(label=f"i{input_idx}")
            input_gene.set_weight(random.uniform(-1.0, 1.0) * 2)  # Stronger weights
            new_genes.append(input_gene)
        genome.next_id += 1
        genome.genes[parent_idx+1:parent_idx+1] = new_genes
        parent_vertex.arity += 1
    
    def _remove_jumper_connection(self, genome: Genome, vertex_gene: VertexGene) -> None:
        """Remove a connection from a vertex"""
        if vertex_gene.arity <= 1:
            return  
        vertex_idx = None
        for i, gene in enumerate(genome.genes):
            if isinstance(gene, VertexGene) and gene.id == vertex_gene.id:
                vertex_idx = i
                break
        if vertex_idx is None:
            return
        jumper_indices = []
        i = vertex_idx + 1
        count = 0
        while i < len(genome.genes) and count < vertex_gene.arity:
            if isinstance(genome.genes[i], (ForwardJumperGene, RecurrentJumperGene, InputGene)):
                jumper_indices.append(i)
                count += 1
            elif isinstance(genome.genes[i], VertexGene):
                break
            i += 1
        if not jumper_indices:
            return  
        remove_idx = random.choice(jumper_indices)
        genome.genes.pop(remove_idx)
        vertex_gene.arity -= 1
        
    def _get_structure_signature(self, genome: Genome) -> str:
        """Get a string signature representing the network structure"""
        vertex_count = sum(1 for gene in genome.genes if isinstance(gene, VertexGene))
        input_count = sum(1 for gene in genome.genes if isinstance(gene, InputGene))
        forward_jumper_count = sum(1 for gene in genome.genes if isinstance(gene, ForwardJumperGene))
        recurrent_jumper_count = sum(1 for gene in genome.genes if isinstance(gene, RecurrentJumperGene))
        return f"V{vertex_count}I{input_count}JF{forward_jumper_count}JR{recurrent_jumper_count}" 
     
    def should_explore(self) -> bool:
        """
        Determine if we should explore new structures based on fitness improvement.
        Returns True if improvement is below threshold (need to explore).
        """
        # Need at least buffer_length+1 generations of history
        if len(self.fitness_history) <= self.buffer_length:
            return False
        # Calculate improvement over the buffer period
        current_fitness = self.fitness_history[-1]
        previous_fitness = self.fitness_history[-self.buffer_length-1]
        improvement = current_fitness - previous_fitness
        # If improvement is below threshold, initiate exploration
        if improvement < self.improvement_threshold:
            print(f"Improvement ({improvement:.6f}) below threshold ({self.improvement_threshold}). Switching to exploration.")
            return True
        else:
            print(f"Good improvement ({improvement:.6f}). Continuing exploitation.")
            return False
    
    def selection(self, mse_values: List[float]) -> Tuple[List[Genome], List[List[int]]]:
        fitness_values = [-mse for mse in mse_values]
        selected_genomes = []
        selected_tree_boundaries = []
        #ELITISM: First, always include the best genome ever found
        if self.best_genome is not None:
            selected_genomes.append(self.best_genome.copy())
            selected_tree_boundaries.append(self.best_tree_boundaries.copy())
            print("Added best genome through elitism")
        
        # 2. PROTECTION: Include all protected genomes
        for i, (genome, tree_boundaries, timer) in enumerate(zip(
                self.protected_genomes, self.protected_tree_boundaries, self.protection_timers)):
            if timer > 0:
                # Only add if not already added through elitism
                genome_signature = self._get_structure_signature(genome)
                already_included = False
                
                for g in selected_genomes:
                    if self._get_structure_signature(g) == genome_signature:
                        already_included = True
                        break
                        
                if not already_included:
                    selected_genomes.append(genome.copy())
                    selected_tree_boundaries.append(tree_boundaries.copy())
                    self.protection_timers[i] -= 1
                    print(f"Protected genome remains for {self.protection_timers[i]} more generations")
        # 3. DIVERSITY: Group genomes by structure (clustering)
        clusters = {}
        for i, genome in enumerate(self.population):
            signature = self._get_structure_signature(genome)
            if signature not in clusters:
                clusters[signature] = []
            clusters[signature].append((i, genome, fitness_values[i]))  # Store index, genome, and fitness
        
        print(f"Found {len(clusters)} different structural clusters")
        
        # Select the best genome from each structural cluster
        for signature, genomes in clusters.items():
            already_included = False
            for g in selected_genomes:
                if self._get_structure_signature(g) == signature:
                    already_included = True
                    break
                    
            if not already_included and len(selected_genomes) < self.population_size:
                # Get the best genome from this cluster
                best_idx, best_genome, _ = max(genomes, key=lambda x: x[2])  # Sort by fitness
                
                # Add to selected population if we have space
                selected_genomes.append(best_genome.copy())
                selected_tree_boundaries.append(self.tree_boundaries_list[best_idx].copy())
        
        # 4. FITNESS: If we still have space, fill remaining slots with best genomes overall
        if len(selected_genomes) < self.population_size:
            sorted_indices = sorted(range(len(self.population)), 
                                key=lambda i: fitness_values[i], 
                                reverse=True)
            
            # Add best genomes that aren't already selected
            for idx in sorted_indices:
                genome = self.population[idx]
                
                # Check if this genome is already included (simple object identity check)
                already_included = False
                for g in selected_genomes:
                    if id(genome) == id(g):  # Simple identity check
                        already_included = True
                        break
                        
                # Add if not already included and we have space
                if not already_included and len(selected_genomes) < self.population_size:
                    selected_genomes.append(genome.copy())
                    selected_tree_boundaries.append(self.tree_boundaries_list[idx].copy())
        
        # Ensure we don't exceed population size (should never happen with proper checks above)
        if len(selected_genomes) > self.population_size:
            selected_genomes = selected_genomes[:self.population_size]
            selected_tree_boundaries = selected_tree_boundaries[:self.population_size]
        
        # Print structure diversity information
        signatures = [self._get_structure_signature(g) for g in selected_genomes]
        unique_signatures = set(signatures)
        print(f"Population has {len(unique_signatures)} different structures")
        
        return selected_genomes, selected_tree_boundaries

    def evolve(self, X: List[List[float]], y: List[float], generations: int = 20) -> Genome:
        self.X = X
        self.y = y
        self.initialize_minimal_population()
        self.fitness_history = []
        self.protected_genomes = []
        self.protected_tree_boundaries = []
        self.protection_timers = []
        protection_period = 3
        for generation in range(generations):
            print(f"\n{'='*20} Generation {generation+1}/{generations} {'='*20}")
            
            # Evaluate population
            print("Evaluating population...")
            mse_values = self.evaluate_population(X, y)
            current_best_fitness = -min(mse_values)
            
            # Update best genome and fitness history
            best_idx = mse_values.index(min(mse_values))
            if not self.best_genome or current_best_fitness > self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_genome = self.population[best_idx].copy()
                self.best_tree_boundaries = self.tree_boundaries_list[best_idx].copy()
                print(f"New best fitness: {self.best_fitness:.6f}")
            else:
                print(f"Best fitness so far: {self.best_fitness:.6f}")
            
            # Add current best fitness to history
            self.fitness_history.append(current_best_fitness)
            
            # Skip mutation on the last generation
            if generation == generations - 1:
                print("Maximum generations reached")
                break
            if not self.should_explore():  
                print("\nStructural Exploitation: Parametric mutation")
                
                # Perform parametric mutation based on fitness-proportional clusters
                fitness_values = {genome: -mse for genome, mse in zip(self.population, mse_values)}
                mutated_population = self.parametric_mutation(self.population, fitness_values)
                
                # Set as current population for evaluation
                self.population = mutated_population
                self.tree_boundaries_list = [genome.find_tree_boundaries() for genome in self.population]
                
                # Evaluate and select from the parametrically mutated population
                mutated_mse = self.evaluate_population(X, y)
                self.population, self.tree_boundaries_list = self.selection(mutated_mse)
            
            else: 
                print("\nStructural Exploration: Creating new individuals through structural mutation")
                new_population = []
                new_tree_boundaries = []
                
                # Apply structural mutation to each genome
                for i, genome in enumerate(self.population):
                    old_signature = self._get_structure_signature(genome)
                    mutated_genome = self.structural_mutation(genome)
                    new_signature = self._get_structure_signature(mutated_genome)
                    mutated_boundaries = mutated_genome.find_tree_boundaries()
                    
                    # If structure changed, protect it
                    if new_signature != old_signature:
                        self.protected_genomes.append(mutated_genome)
                        self.protected_tree_boundaries.append(mutated_boundaries)
                        self.protection_timers.append(protection_period)
                        print(f"New structure discovered! Protected for {protection_period} generations: {new_signature}")
                    
                    new_population.append(mutated_genome)
                    new_tree_boundaries.append(mutated_boundaries)
                
                # Update population with structurally mutated genomes
                self.population = new_population
                self.tree_boundaries_list = new_tree_boundaries
                
                # Evaluate and select from the structurally mutated population
                structural_mse = self.evaluate_population(X, y)
                self.population, self.tree_boundaries_list = self.selection(structural_mse)
            
            # Print population structure diversity
            signatures = [self._get_structure_signature(g) for g in self.population]
            unique_signatures = set(signatures)
            print(f"Population has {len(unique_signatures)} different structures")
            for sig in unique_signatures:
                count = signatures.count(sig)
                print(f"  {sig}: {count} genomes")
        
        print(f"\nEvolution complete. Best fitness achieved: {self.best_fitness:.6f}")
        return self.best_genome