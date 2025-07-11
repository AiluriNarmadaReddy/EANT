from typing import List, Dict, Tuple
from activation_util import ActivationUtil
from genes import VertexGene, InputGene, ForwardJumperGene, RecurrentJumperGene, Gene
from genome import Genome

class NeuralNetworkEvaluator:
    def __init__(self, genome: Genome, tree_boundaries: List[Tuple[int, int]]):
        self.genome = genome
        self.tree_boundaries = tree_boundaries
        
    def evaluate(self, inputs: Dict[str, float]) -> List[float]:
        outputs = []
        
        for start, end in self.tree_boundaries:
            tree_genes = self.genome.genes[start:end]
            output = self._evaluate_tree(tree_genes, inputs)
            outputs.append(output)
            
        return outputs
    
    def _evaluate_tree(self, genes: List[Gene], input_values: Dict[str, float]) -> float:
        """Evaluate a single tree in prefix order."""
        idx = 0
        
        def process_gene() -> float:
            nonlocal idx
            if idx >= len(genes):
                raise ValueError("Premature end of gene sequence")
            
            gene = genes[idx]
            idx += 1
            
            if isinstance(gene, InputGene):
                input_value = input_values.get(gene.label, 0.0)
                output = input_value * gene.get_weight()
                gene.set_current_output(output)
                return output
                
            elif isinstance(gene, ForwardJumperGene):
                for g in self.genome.genes:
                    if isinstance(g, VertexGene) and g.id == gene.source:
                        jumper_output = g.get_current_output() * gene.get_weight()
                        gene.set_current_output(jumper_output)
                        return jumper_output
                return 0.0
                
            elif isinstance(gene, RecurrentJumperGene):
                for g in self.genome.genes:
                    if isinstance(g, VertexGene) and g.id == gene.source:
                        jumper_output = g.get_current_output() * gene.get_weight()
                        gene.set_current_output(jumper_output)
                        return jumper_output
                return 0.0
                
            elif isinstance(gene, VertexGene):
                input_list = [] 
                for _ in range(gene.arity):
                    input_value = process_gene() 
                    input_list.append(input_value)
                
                activation_fn = getattr(ActivationUtil, gene.activation)
                weighted_sum = sum(input_list)
                output = activation_fn(weighted_sum) * gene.get_weight()
                gene.set_current_output(output)
                return output
                
            else:
                raise ValueError(f"Unknown gene type: {gene}")
        
        return process_gene()
    
    def save_state(self) -> Dict[int, float]:
        """Save the current state of all neurons for recurrent connections."""
        state = {}
        for gene in self.genome.genes:
            if isinstance(gene, VertexGene):
                state[gene.id] = gene.get_current_output()
        return state
    
    def restore_state(self, state: Dict[int, float]) -> None:
        """Restore neuron states from a saved state."""
        for gene in self.genome.genes:
            if isinstance(gene, VertexGene) and gene.id in state:
                gene.set_current_output(state[gene.id])
                
    def reset_state(self) -> None:
        """Reset all computation states to zero."""
        for gene in self.genome.genes:
            gene.set_current_output(0.0)