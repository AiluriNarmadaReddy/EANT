import random
from typing import List, Tuple, Dict, Optional, Set
from genes import VertexGene, InputGene, JumperGene, ForwardJumperGene, RecurrentJumperGene, Gene

class ExpressionNode:
    """
    Node for building an expression tree in memory before flattening.
    """
    def __init__(self, gene):
        self.gene = gene
        self.children: List['ExpressionNode'] = []

class Genome:
    def __init__(self, depth: int = 1, genes=None):
        self.genes = genes if genes is not None else []
        self.next_id = self._calculate_next_id()
        self.depth = depth
        
    def _calculate_next_id(self) -> int:
        if not self.genes:
            return 0
        return max([g.id for g in self.genes if isinstance(g, VertexGene) and hasattr(g, 'id')], default=-1) + 1
        
    def build_minimal_tree(self, input_count: int, output_count: int) -> List[Tuple[int, int]]:
        self.genes = []
        tree_boundaries = []
        next_vertex_id = 0
        
        for o in range(output_count):
            start = len(self.genes)
            
            # Create output neuron (in prefix notation, neuron comes first)
            neuron = VertexGene(gene_id=next_vertex_id, arity=0)  # Start with 0 arity, will update later
            next_vertex_id += 1
            self.genes.append(neuron)
            
            # Randomly select ~50% of inputs
            num_connections = max(1, input_count // 2)  # At least 1 connection
            selected_inputs = random.sample(range(input_count), num_connections)
            
            # Update neuron's arity
            neuron.arity = len(selected_inputs)
            
            # Add the selected input genes (in prefix notation, inputs come after the neuron)
            for input_idx in selected_inputs:
                input_gene = InputGene(label=f"i{input_idx}")
                self.genes.append(input_gene)
            
            end = len(self.genes)
            tree_boundaries.append((start, end))
        
        # Update next_id
        self.next_id = next_vertex_id
        
        # Validate the resulting genome
        valid, error = self.is_valid()
        if not valid:
            raise ValueError(f"Failed to create valid genome: {error}")
        
        return tree_boundaries
        
    def build_expression_tree(self, 
                         input_count: int, 
                         output_count: int, 
                         jumper_prob: float = None) -> List[Tuple[int, int]]:
        # Set default jumper probability based on depth
        if jumper_prob is None:
            jumper_prob = 0.2 if self.depth > 1 else 0
        
        # For minimal trees, use the simple method
        if self.depth <= 1:
            return self.build_minimal_tree(input_count, output_count)
        
        self.genes = []
        tree_boundaries = []
        next_vertex_id = 0
        
        def build_tree(current_depth: int, available_ids: Set[int]) -> ExpressionNode:
            nonlocal next_vertex_id
            
            # Base case: leaves or local jumpers
            if current_depth == self.depth:
                if available_ids and current_depth > 1 and random.random() < jumper_prob:
                    src = random.choice(list(available_ids))
                    gene = (ForwardJumperGene(src)
                            if random.random() < 0.7
                            else RecurrentJumperGene(src))
                else:
                    idx = random.randrange(input_count)
                    gene = InputGene(label=f"i{idx}")
                return ExpressionNode(gene)
            
            # Create a vertex gene with random arity [1..input_count]
            arity = random.randint(1, min(input_count, 3))
            gene = VertexGene(gene_id=next_vertex_id, arity=arity)
            this_id = next_vertex_id
            next_vertex_id += 1
            
            node = ExpressionNode(gene)
            available_ids.add(this_id)
            
            # Create a dictionary to track input weights for this vertex
            # This ensures that inputs with the same label use the same weight
            input_weights = {}
            
            for _ in range(arity):
                if current_depth < self.depth - 1 and available_ids and random.random() < jumper_prob:
                    unused = available_ids - {this_id}
                    if unused:
                        src = random.choice(list(unused))
                        jmp = (ForwardJumperGene(src)
                              if random.random() < 0.7
                              else RecurrentJumperGene(src))
                        node.children.append(ExpressionNode(jmp))
                        continue
                
                # Otherwise, create a subtree
                child = build_tree(current_depth + 1, available_ids)
                
                # Ensure consistent weights for input genes with the same label
                if isinstance(child.gene, InputGene):
                    label = child.gene.label
                    if label in input_weights:
                        # If this input label has been used before, use the same weight
                        child.gene.set_weight(input_weights[label])
                    else:
                        # Otherwise, store the randomly generated weight
                        input_weights[label] = child.gene.get_weight()
                
                node.children.append(child)
            
            return node
        
        def flatten(node: ExpressionNode, out: List):
            out.append(node.gene)
            for c in node.children:
                flatten(c, out)
        
        # Build each tree and flatten
        for _ in range(output_count):
            available_ids: Set[int] = set()
            root = build_tree(0, available_ids)
            flat: List = []
            flatten(root, flat)
            start = len(self.genes)
            self.genes.extend(flat)
            end = len(self.genes)
            tree_boundaries.append((start, end))
        
        # Update next_id
        self.next_id = next_vertex_id
        
        # Validate the resulting genome
        valid, error = self.is_valid()
        if not valid:
            raise ValueError(f"Failed to create valid genome: {error}")
        
        return tree_boundaries
        
    def calculate_v_values(self):
        """Calculate v(xi) values: 1-arity for vertex genes, 1 for other genes"""
        v_values = []
        for gene in self.genes:
            if isinstance(gene, VertexGene):
                v_values.append(1 - gene.arity)
            else:
                v_values.append(1)
        return v_values
        
    def calculate_s_values(self):
        """Calculate s values (running sum of v values)"""
        v_values = self.calculate_v_values()
        s_values = [0]  # s1 = 0
        for v in v_values:
            s_values.append(s_values[-1] + v)
        return s_values
        
    def find_tree_boundaries(self) -> List[Tuple[int, int]]:
        """
        Identify the boundaries of all trees in the genome.
        
        Returns:
            List of (start_index, end_index) tuples for each tree
        """
        v_values = self.calculate_v_values()
        tree_boundaries = []
        start = 0
        
        while start < len(self.genes):
            # Find the end of the current tree
            integer_sum = 0
            end = start
            
            while end < len(self.genes):
                integer_sum += v_values[end]
                end += 1
                
                if integer_sum == 1:
                    # Found a complete tree
                    tree_boundaries.append((start, end))
                    break
            
            # If we reached the end without finding a complete tree,
            # do NOT add it as a tree - it's an invalid structure
            if end >= len(self.genes) and integer_sum != 1:
                return []  # Return empty list to indicate invalid genome
            
            # Move to the next tree
            start = end
            
        return tree_boundaries
        
    def is_valid(self) -> Tuple[bool, str]:
        """
        Check if the genome is valid by validating all trees.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # First, find all tree boundaries
        tree_boundaries = self.find_tree_boundaries()
        
        if not tree_boundaries:
            return False, "No valid trees found in the genome or incomplete trees present"
        
        # Ensure all genes are part of some tree
        covered_indices = set()
        for start, end in tree_boundaries:
            for i in range(start, end):
                covered_indices.add(i)
        
        if len(covered_indices) < len(self.genes):
            return False, "Not all genes are part of a valid tree"
        
        # Then validate each tree
        for i, (start, end) in enumerate(tree_boundaries):
            subtree = self.genes[start:end]
            valid, error = self._validate_tree(subtree, start_idx=start)
            if not valid:
                return False, f"Tree {i+1} (genes[{start}:{end}]) is invalid: {error}"
        
        return True, "Genome is valid"
    
    def _validate_tree(self, genes: List, start_idx: int = 0) -> Tuple[bool, str]:
        """
        Validate a prefix-encoded expression tree.
        Ensures each vertex has correct arity and jumpers refer locally.

        Returns:
            (True, "") if valid, else (False, error_message).
        """
        subtree_vertices = {g.id for g in genes if isinstance(g, VertexGene)}
        idx = 0

        def walk() -> Tuple[bool, str]:
            nonlocal idx
            if idx >= len(genes):
                return False, f"Premature end at token {idx}"  # ran out of tokens

            g = genes[idx]
            pos = start_idx + idx
            idx += 1

            if isinstance(g, VertexGene):
                # Check that vertex has at least one input
                if g.arity < 1:
                    return False, f"Vertex at pos {pos} has arity {g.arity}, must be at least 1"
                
                for _ in range(g.arity):
                    valid, err = walk()
                    if not valid:
                        return False, err
                    token = genes[idx - 1]
                    # Jumper must refer within this subtree
                    if isinstance(token, JumperGene):
                        if token.source not in subtree_vertices:
                            return False, (
                                f"Jumper at pos {pos+1} refs outside subtree: {token.source}"
                            )
                return True, ""

            elif isinstance(g, (InputGene, ForwardJumperGene, RecurrentJumperGene)):
                return True, ""

            else:
                return False, f"Unknown gene at pos {pos}: {g}"

        valid, error = walk()
        if not valid:
            return False, error
        if idx != len(genes):
            return False, f"Consumed {idx}/{len(genes)} tokens"
        return True, ""
    
    def get_tree_info(self) -> str:
        """
        Get detailed information about each tree in the genome.
        
        Returns:
            String with tree information
        """
        tree_boundaries = self.find_tree_boundaries()
        info = []
        
        # Map vertex IDs to global positions
        vertex_pos = {
            g.id: i for i, g in enumerate(self.genes) if isinstance(g, VertexGene)
        }
        
        for idx, (start, end) in enumerate(tree_boundaries):
            subtree = self.genes[start:end]
            info.append(f"Tree {idx+1} (genes[{start}:{end}]):")
            
            # Detailed representation
            for gene in subtree:
                info.append(f"  {gene}")
            
            # Compact representation
            info.append(f"  Compact: {' '.join(g.short_repr() for g in subtree)}")
            
            # Jumper cross-references
            jumpers = [g for g in subtree if isinstance(g, JumperGene)]
            if jumpers:
                info.append("  Jumpers:")
                for j in jumpers:
                    global_pos = vertex_pos.get(j.source, None)
                    tree_idx = next((i for i, (s, e) in enumerate(tree_boundaries)
                                     if global_pos is not None and s <= global_pos < e), None)
                    info.append(f"    {j.short_repr()} -> Vertex {j.source} at global pos {global_pos}, tree {tree_idx+1 if tree_idx is not None else 'N/A'}")
            
            # Validation
            valid, err = self._validate_tree(subtree, start_idx=start)
            info.append(f"  Valid: {'Yes' if valid else 'No - ' + err}")
            info.append("")
        
        return "\n".join(info)
        
    def __str__(self):
        return " ".join(str(gene) for gene in self.genes)
        
    def compact_repr(self):
        """Return a compact representation of the genome"""
        return " ".join(gene.short_repr() for gene in self.genes)
        
    def copy(self) -> 'Genome':
        """Create a deep copy of this genome."""
        new_genome = Genome(self.depth)
        new_genome.next_id = self.next_id
        new_genome.genes = [gene.copy() for gene in self.genes]
        return new_genome
    
    def calculate_tree_boundaries(self) -> List[Tuple[int, int]]:
        return self.find_tree_boundaries()
        
    def get_vertex_genes(self) -> List[VertexGene]:
        """Return all vertex genes in the genome."""
        return [g for g in self.genes if isinstance(g, VertexGene)]
        
    def get_input_genes(self) -> List[InputGene]:
        """Return all input genes in the genome."""
        return [g for g in self.genes if isinstance(g, InputGene)]
        
    def get_jumper_genes(self) -> List[JumperGene]:
        """Return all jumper genes in the genome."""
        return [g for g in self.genes if isinstance(g, JumperGene)]
        
    def get_used_inputs(self) -> Set[str]:
        """Return a set of all input labels used in the genome."""
        return {g.label for g in self.genes if isinstance(g, InputGene)}
        
    def get_vertex_map(self) -> Dict[int, VertexGene]:
        """Return a mapping from vertex IDs to vertex genes."""
        return {g.id: g for g in self.genes if isinstance(g, VertexGene)}
        
    def get_vertex_depth(self, vertex_id: int) -> int:
        # Calculate s values
        s_values = self.calculate_s_values()
        
        # Find the vertex
        vertex_index = None
        for i, gene in enumerate(self.genes):
            if isinstance(gene, VertexGene) and gene.id == vertex_id:
                vertex_index = i
                break
        
        if vertex_index is None:
            return -1  # Vertex not found
        
        # Calculate depth
        depth = 0
        parent_index = None
        
        # Find parent vertex
        for i in range(vertex_index - 1, -1, -1):
            if isinstance(self.genes[i], VertexGene):
                # Check if this is the parent
                if s_values[i] >= s_values[vertex_index+1] and all(s_values[k] < s_values[vertex_index+1] for k in range(i+1, vertex_index)):
                    parent_index = i
                    break
        
        # If no parent found, this is an output vertex
        if parent_index is None:
            return 0
        
        # Recursively get parent depth
        return 1 + self.get_vertex_depth(self.genes[parent_index].id)