try:
    import shapely.geometry as sg
except ImportError:
    sg = None

try:
    import geopandas as gp
except Exception:
    # Catch broad exception to handle the case of a corrupt geopandas installation
    gp = None
