from core.tasks.dog.dog import Dog
from core.tasks.kangaroo.kangaroo import Kangaroo
from core.tasks.raptor.raptor import Raptor

# Mappings from strings to environments
task_map = {
    "Dog": Dog,
    "Raptor": Raptor,
    "Kangaroo": Kangaroo,
}
