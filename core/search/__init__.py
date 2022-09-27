from .fix import Fix
from .pso_search.pso_search import PSOSearch
from .random_search import RSearch

search_map = {
    'Fix': Fix,
    'RSearch': RSearch,
    'PSOSearch': PSOSearch
}
