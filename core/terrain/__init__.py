from .flat import FlatTerrain, Flat2Terrain
from .slope import UphillTerrain, DownhillTerrain
from .tunnel import TunnelTerrain

terrain_map = {
    'Flat': FlatTerrain,
    'Uphill': UphillTerrain,
    'Downhill': DownhillTerrain,
    'Tunnel': TunnelTerrain,
    'Flat2': Flat2Terrain,
}
