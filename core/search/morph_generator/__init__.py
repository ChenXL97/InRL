from ._parse_morph_tensor import parse_morph_tensor
from .dog_mg import DogMorphGenerator
from .kangaroo_mg import KangarooMorphGenerator
from .raptor_mg import RaptorMorphGenerator

# Mappings from strings to morph generators
morph_generator_map = {
    "Dog": DogMorphGenerator,
    "Raptor": RaptorMorphGenerator,
    "Kangaroo": KangarooMorphGenerator,
}
