import numpy as np
from numba import jitclass
from numba import int64, float64

from spatialpandas.spatialindex.hilbert_curve import (
    _data2coord, distance_from_coordinates
)
from spatialpandas.utils import ngjit


@ngjit
def left_child(i):
    return 2 * i + 1


@ngjit
def right_child(i):
    return 2 * i + 2


@ngjit
def parent(i):
    return (i - 1) // 2


spec = [
    ('p', int64),
    ('input_size', int64),
    ('tree_length', int64),
    ('tree_depth', int64),
    ('leaf_start', int64),
    ('bounds', float64[:, :]),
    ('bounds_tree', float64[:, :]),
    ('hilbert_distances', int64[:]),
    ('sorting_indices', int64[:]),
]


@jitclass(spec)
class HilbertRtree(object):
    """
    This class is a numba implementation of a read-only Hilbert R-tree spatial index

    See https://en.wikipedia.org/wiki/Hilbert_R-tree for more info on the Hilbert R-tree

    This implementation stores the R-tree as an array representation of a binary tree.
    See https://en.wikipedia.org/wiki/Binary_tree#Arrays for more info on the array
    representation of a binary tree.
    """
    def __init__(self, bounds, p=10):
        """
        Parameters
        ----------
        bounds: array
            A 2-dimensional numpy array representing a collection of bounding boxes.
            One row per bounding box with 4 columns containing the coordinates of each
            bounding box as follows:
                 - bounds[:, 0] is min x-coordinate for each bounding box
                 - bounds[:, 1] is min y-coordinate for each bounding box
                 - bounds[:, 2] is max x-coordinate for each bounding box
                 - bounds[:, 3] is max y-coordinate for each bounding box
        p: int (default 10)
            The Hilbert curve order parameter that determines the resolution
            of the 2D grid that data points are rounded to before computing
            their Hilbert distance. Points will be discretized into 2 ** p
            bins in each the x and y dimensions.
        """
        # Validate bounds
        if len(bounds.shape) != 2:
            raise ValueError("bounds must be a 2D array")

        if bounds.shape[0] == 0:
            raise ValueError("The first dimension of bounds must not be empty")

        if bounds.shape[1] != 4:
            raise ValueError("The second dimension of bounds must length 4")

        # Hilbert iterations
        self.p = p

        # Init bounds_tree array for storing the binary tree representation
        self.input_size = bounds.shape[0]
        self.tree_depth = int(np.ceil(np.log2(bounds.shape[0])))
        next_pow2 = 2 ** self.tree_depth
        self.tree_length = next_pow2 * 2 - 1
        self.bounds = bounds
        self.bounds_tree = np.full((self.tree_length, 4), np.nan)
        self.leaf_start = self.tree_length - next_pow2

        # Compute Hilbert distances for inputs
        side_length = 2 ** p
        x_range = (bounds[:, 0].min(), bounds[:, 2].max())
        y_range = (bounds[:, 1].min(), bounds[:, 3].max())

        # Avoid degenerate case where there is a single unique rectangle that is a
        # single point. Increase the range by 1.0 to prevent divide by zero error
        if x_range[0] == x_range[1]:
            x_range = (x_range[0], x_range[0] + 1.0)
        if y_range[0] == y_range[1]:
            y_range = (y_range[0], y_range[0] + 1.0)

        # Compute hilbert distance of the middle of each bounding box
        x_mids = (bounds[:, 0] + bounds[:, 2]) / 2.0
        y_mids = (bounds[:, 1] + bounds[:, 3]) / 2.0
        x_coords = _data2coord(x_mids, x_range, side_length)
        y_coords = _data2coord(y_mids, y_range, side_length)
        self.hilbert_distances = distance_from_coordinates(p, x_coords, y_coords)

        # Calculate indices needed to sort bounds by hilbert distance
        self.sorting_indices = np.argsort(self.hilbert_distances)

        # Populate leaves of the tree. This is layer = tree_depth
        self.bounds_tree[self.leaf_start:self.leaf_start + bounds.shape[0]] = \
            bounds[self.sorting_indices, :]

        # Populate internal layers of tree
        layer = self.tree_depth - 1
        start = parent(self.tree_length - next_pow2)
        stop = parent(self.tree_length - 1)

        while layer >= 0:
            for node in range(start, stop + 1):
                left_bounds = self.bounds_tree[left_child(node), :]
                left_valid = not np.isnan(left_bounds[0])
                right_bounds = self.bounds_tree[right_child(node), :]
                right_valid = not np.isnan(right_bounds[0])

                if left_valid:
                    if right_valid:
                        xmin = min(left_bounds[0], right_bounds[0])
                        ymin = min(left_bounds[1], right_bounds[1])
                        xmax = max(left_bounds[2], right_bounds[2])
                        ymax = max(left_bounds[3], right_bounds[3])
                        self.bounds_tree[node, :] = np.array([xmin, ymin, xmax, ymax],
                                                             dtype=np.float64)
                    else:
                        self.bounds_tree[node, :] = left_bounds

                elif right_valid:
                    self.bounds_tree[node, :] = right_bounds

            # update layer, start/stop
            start = parent(start)
            stop = parent(stop)
            layer -= 1

    def _start_index(self, node):
        """
        The first index into sorting_indices represented by node
        """
        while True:
            child = left_child(node)
            if child >= self.tree_length:
                return node - self.leaf_start
            else:
                node = child

    def _stop_index(self, node):
        """
        One past the last index into sorting_indices represented by node
        """
        while True:
            child = right_child(node)
            if child >= self.tree_length:
                return node - self.leaf_start + 1
            else:
                node = child

    def intersection(self, query_bounds, result):
        """
        Compute the indices of the input bounding boxes that intersect with the
        supplied query bounds

        Parameters
        ----------
        query_bounds
            A 4 element array of the form [xmin, ymin, xmax, ymax] representing the
            bounding for which intersections should be computed
        result
            integer array that the indices of the intersecting bounding boxes
            should be populated into

        Returns
        -------
        int
            The number of intersecting bounding boxes
        """
        if len(query_bounds) != 4:
            raise ValueError('query_bounds must be a 4 element array')

        if len(result) < self.input_size:
            raise ValueError("""
The supplied result array must have at least as many elements as the input""")

        x0, y0, x1, y1 = query_bounds

        nodes = [0]

        # Intersection of ranges below page_size are computed by brute force
        page_size = 2 ** 9
        result_start = 0

        # Find ranges of indices that overlap with query bounds
        while nodes:
            next_node = nodes.pop()
            node_x0, node_y0, node_x1, node_y1 = self.bounds_tree[next_node, :]

            if x1 < node_x0 or x0 > node_x1 or y1 < node_y0 or y0 > node_y1:
                # Node's bounds do not overlap with query bounds
                pass
            elif node_x0 >= x0 and node_x1 <= x1 and node_y0 >= y0 and node_y1 <= y1:
                # Node's bounds are fully inside query bounds
                start = self._start_index(next_node)
                stop = self._stop_index(next_node)
                next_slice = self.sorting_indices[start:stop]
                result[result_start:result_start + len(next_slice)] = next_slice
                result_start += len(next_slice)
            else:
                start = self._start_index(next_node)
                stop = self._stop_index(next_node)
                if stop - start == 1:
                    # leaf
                    result[result_start] = self.sorting_indices[start]
                    result_start += 1
                elif stop - start < page_size:
                    next_slice = self.sorting_indices[start:stop]
                    bounds_slice = self.bounds[next_slice, :]
                    outside = ((bounds_slice[:, 2] < x0) | (bounds_slice[:, 0] > x1) |
                               (bounds_slice[:, 3] < y0) | (bounds_slice[:, 1] > y1))

                    next_slice = next_slice[~outside]
                    result[result_start:result_start + len(next_slice)] = next_slice
                    result_start += len(next_slice)
                else:
                    # Partial overlap of interior bounding box, recurse to children
                    nodes.extend([right_child(next_node), left_child(next_node)])

        # Return start index
        return result_start
