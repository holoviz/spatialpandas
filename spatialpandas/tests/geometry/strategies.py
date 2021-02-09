import numpy as np
from geopandas import GeoSeries
from geopandas.array import from_shapely
from hypothesis import HealthCheck, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from scipy.spatial.qhull import Voronoi
from shapely import geometry as sg
from shapely.affinity import scale, translate
from shapely.ops import cascaded_union, polygonize

hyp_settings = settings(
    deadline=None,
    max_examples=500,
    suppress_health_check=[HealthCheck.too_slow],
)

coord = st.floats(
    allow_infinity=False, allow_nan=False, max_value=1000, min_value=-1000
)

st_points = arrays(
    elements=st.floats(
        allow_infinity=False, allow_nan=False, max_value=60, min_value=-60
    ),
    shape=(100, 2),
    dtype='float64'
)


@st.composite
def st_point_array(draw, min_size=0, max_size=30, geoseries=False):
    n = draw(st.integers(min_size, max_size))
    points = []
    for i in range(n):
        x_mid = draw(st.floats(-50, 50))
        y_mid = draw(st.floats(-50, 50))
        point = (np.random.rand(2) - 0.5) * 5
        point[0] = point[0] + x_mid
        point[1] = point[1] + y_mid
        points.append(sg.Point(point))

    result = from_shapely(points)
    if geoseries:
        result = GeoSeries(result)
    return result


@st.composite
def st_multipoint_array(draw, min_size=0, max_size=30, geoseries=False):
    n = draw(st.integers(min_size, max_size))
    lines = []
    for i in range(n):
        num_points = draw(st.integers(1, 10))
        x_mid = draw(st.floats(-50, 50))
        y_mid = draw(st.floats(-50, 50))
        points = (np.random.rand(num_points, 2) - 0.5) * 5
        points[:, 0] = points[:, 0] + x_mid
        points[:, 1] = points[:, 1] + y_mid
        lines.append(sg.MultiPoint(points))

    result = from_shapely(lines)
    if geoseries:
        result = GeoSeries(result)
    return result


@st.composite
def st_line_array(draw, min_size=0, max_size=30, geoseries=False):
    n = draw(st.integers(min_size, max_size))
    lines = []
    for i in range(n):
        line_len = draw(st.integers(2, 10))
        x_mid = draw(st.floats(-50, 50))
        y_mid = draw(st.floats(-50, 50))
        points = np.cumsum(np.random.rand(line_len, 2) - 0.5, axis=0)
        points[:, 0] = points[:, 0] + x_mid
        points[:, 1] = points[:, 1] + y_mid
        lines.append(sg.LineString(points))

    result = from_shapely(lines)
    if geoseries:
        result = GeoSeries(result)
    return result


def get_unique_points(
        n, x_range=(0, 10), x_grid_dim=101, y_range=(0, 10), y_grid_dim=101
):
    """
    Get array of unique points, randomly drawn from a uniform grid
    """
    xs, ys = np.meshgrid(
        np.linspace(x_range[0], x_range[1], x_grid_dim),
        np.linspace(y_range[0], y_range[1], y_grid_dim),
    )
    points = np.stack([xs.flatten(), ys.flatten()], axis=1)
    selected_inds = np.random.choice(np.arange(points.shape[0]), n, replace=False)
    return points[selected_inds, :]


@st.composite
def st_ring_array(draw, min_size=3, max_size=30, geoseries=False):
    assert min_size >= 3
    n = draw(st.integers(min_size, max_size))
    rings = []
    for i in range(n):
        rings.append(sg.LinearRing(get_unique_points(n)))

    result = from_shapely(rings)
    if geoseries:
        result = GeoSeries(result)
    return result


@st.composite
def st_multiline_array(draw, min_size=0, max_size=5, geoseries=False):
    n = draw(st.integers(min_size, max_size))
    multilines = []
    for i in range(n):
        m = draw(st.integers(1, 5))
        lines = []
        for j in range(m):
            line_len = draw(st.integers(2, 3))
            x_mid = draw(st.floats(-50, 50))
            y_mid = draw(st.floats(-50, 50))
            points = np.cumsum(np.random.rand(line_len, 2) - 0.5, axis=0)
            points[:, 0] = points[:, 0] + x_mid
            points[:, 1] = points[:, 1] + y_mid
            lines.append(sg.LineString(points))
        multilines.append(sg.MultiLineString(lines))

    result = from_shapely(multilines)
    if geoseries:
        result = GeoSeries(result)
    return result


@st.composite
def st_polygon(draw, n=10, num_holes=None, xmid=0, ymid=0):
    # Handle defaults
    if num_holes is None:
        num_holes = draw(st.integers(0, 4))

    # Build outer shell
    tries = 50
    poly = None
    while tries > 0:
        tries -= 1
        points = (np.random.rand(n, 2) - 0.5) * 100
        points[:, 0] = points[:, 0] + xmid
        points[:, 1] = points[:, 1] + ymid

        vor = Voronoi(points)
        mls = sg.MultiLineString([
            vor.vertices[s] for s in vor.ridge_vertices if all(np.array(s) >= 0)
        ])

        poly = cascaded_union(list(polygonize(mls)))
        poly = poly.intersection(sg.box(-50, -50, 50, 50))
        if isinstance(poly, sg.Polygon) and not poly.is_empty:
            break

    if not isinstance(poly, sg.Polygon):
        raise ValueError("Failed to construct polygon")

    # Build holes
    remaining_holes = num_holes
    tries = 50
    while remaining_holes > 0 and tries > 0:
        tries -= 1
        points = (np.random.rand(n, 2) - 0.5) * 50
        points[:, 0] = points[:, 0] + xmid
        points[:, 1] = points[:, 1] + ymid
        vor = Voronoi(points)
        mls = sg.MultiLineString([
            vor.vertices[s] for s in vor.ridge_vertices if all(np.array(s) >= 0)
        ])
        hole_components = [p for p in polygonize(mls) if poly.contains(p)]
        if hole_components:
            hole = cascaded_union([p for p in polygonize(mls) if poly.contains(p)])
            if isinstance(hole, sg.MultiPolygon):
                hole = hole[0]

            new_poly = poly.difference(hole)
            if isinstance(new_poly, sg.Polygon):
                poly = new_poly
                remaining_holes -= 1

    return poly


@st.composite
def st_polygon_array(draw, min_size=0, max_size=5, geoseries=False):
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    sg_polygons = [
        draw(st_polygon(xmid=draw(st.floats(-50, 50)), ymid=draw(st.floats(-50, 50))))
        for _ in range(n)
    ]

    result = from_shapely(sg_polygons)
    if geoseries:
        result = GeoSeries(result)
    return result


@st.composite
def st_multipolygon_array(draw, min_size=0, max_size=5, geoseries=False):
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    sg_multipolygons = []
    for _ in range(n):
        xmid=draw(st.floats(-50, 50))
        ymid=draw(st.floats(-50, 50))
        polygon = draw(st_polygon(xmid=xmid, ymid=ymid))
        m = draw(st.integers(min_value=1, max_value=4))

        # populate polygons with m copies of polygon
        polygons = [polygon] * m

        # translate polygons so they don't overlap
        _, _, last_x1, last_y1 = polygons[0].bounds
        for j in range(1, m):
            polygon = scale(polygons[j], 0.8, 0.5)
            poly_x0, poly_y0, poly_x1, poly_y1 = polygon.bounds
            new_polygon = translate(
                polygon, yoff=last_y1 - poly_y0 + 1
            )
            _, _, last_x1, last_y1 = new_polygon.bounds
            polygons[j] = new_polygon

        sg_multipolygons.append(sg.MultiPolygon(polygons))

    result = from_shapely(sg_multipolygons)
    if geoseries:
        result = GeoSeries(result)
    return result


@st.composite
def st_bounds(draw, x_min=-60, y_min=-60, x_max=60, y_max=60, orient=False):
    x_float = st.floats(
        allow_infinity=False, allow_nan=False, min_value=x_min, max_value=x_max,
    )
    y_float = st.floats(
        allow_infinity=False, allow_nan=False, min_value=y_min, max_value=y_max,
    )
    # Generate x range
    x0, x1 = draw(x_float), draw(x_float)

    # Generate y range
    y0, y1 = draw(y_float), draw(y_float)

    if orient:
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0

    return x0, y0, x1, y1
