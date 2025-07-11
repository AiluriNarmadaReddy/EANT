import random
from enum import Enum

class GeneType(Enum):
    VERTEX = 'V'
    INPUT = 'I' 
    FORWARD_JUMPER = 'JF'
    RECURRENT_JUMPER = 'JR'

class Gene:
    def __init__(self, gene_type: GeneType, weight: float = None):
        self.gene_type = gene_type
        self.weight = weight if weight is not None else random.uniform(-1.0, 1.0)
        self.current_output = 0.0  
    
    def __str__(self):
        return f"{self.gene_type.value}(w={self.weight:.2f})"
    
    def short_repr(self):
        return "?"

    def set_weight(self, new_weight: float) -> None:
        self.weight = new_weight

    def get_weight(self) -> float:
        return self.weight
        
    def set_current_output(self, value: float) -> None:
        """Store the result of current computation."""
        self.current_output = value
        
    def get_current_output(self) -> float:
        """Get the saved result of previous computation."""
        return self.current_output
    
    def copy(self):
        """Create a deep copy of this gene."""
        new_gene = type(self).__new__(type(self))
        new_gene.__dict__.update(self.__dict__)
        return new_gene

class VertexGene(Gene):
    def __init__(self, gene_id: int, arity: int, activation: str = None, weight: float = None):
        super().__init__(GeneType.VERTEX, weight)
        self.id = gene_id
        self.arity = arity
        activation_options = ['relu', 'sigmoid', 'tanh', 'linear', 'leaky_relu']
        self.activation = activation or random.choice(activation_options)
    
    def __str__(self):
        return f"{self.gene_type.value}{self.id}(arity={self.arity}, act={self.activation}, w={self.weight:.2f})"
    
    def short_repr(self):
        return f"v{self.id}"
    
    def copy(self):
        """Create a deep copy of this vertex gene."""
        new_gene = super().copy()
        return new_gene

class InputGene(Gene):
    def __init__(self, label: str, weight: float = None):
        super().__init__(GeneType.INPUT, weight)
        self.label = label
    
    def __str__(self):
        return f"{self.gene_type.value}(label={self.label}, w={self.weight:.2f})"
    
    def short_repr(self):
        return f"{self.label}"

class JumperGene(Gene):
    def __init__(self, gene_type: GeneType, source: int, weight: float = None):
        super().__init__(gene_type, weight)
        self.source = source
    
    def __str__(self):
        return f"{self.gene_type.value}(src={self.source}, w={self.weight:.2f})"
    
    def short_repr(self):
        prefix = "jf" if self.gene_type == GeneType.FORWARD_JUMPER else "jr"
        return f"{prefix}{self.source}"

class ForwardJumperGene(JumperGene):
    def __init__(self, source: int, weight: float = None):
        super().__init__(GeneType.FORWARD_JUMPER, source, weight)
    
    def short_repr(self):
        return f"jf{self.source}"

class RecurrentJumperGene(JumperGene):
    def __init__(self, source: int, weight: float = None):
        super().__init__(GeneType.RECURRENT_JUMPER, source, weight)
    
    def short_repr(self):
        return f"jr{self.source}"