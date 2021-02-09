import numpy as np
from numba import float64, int64

try:
    from numba.experimental import jitclass
except ImportError:
    from numba import jitclass

from ..spatialindex.hilbert_curve import distances_from_coordinates
from ..utils import _data2coord, ngjit


@ngjit
def _left_child(node):
    """
    Args:
        node: index of a binary tree node

    Returns:
        index of node's left child
    """
    return 2 * node + 1


@ngjit
def _right_child(node):
    """
    Args:
        node: index of a binary tree node

    Returns:
        index of node's right child
    """
    return 2 * node + 2


@ngjit
def _parent(node):
    """
    Args:
        node: index of a binary tree node

    Returns:
        index of node's parent
    """
    return (node - 1) // 2


@ngjit
def _distances_from_bounds(bounds, total_bounds, p):
    n = bounds.shape[1] // 2
    dim_ranges = [(total_bounds[d], total_bounds[d + n]) for d in range(n)]
    # Avoid degenerate case where there is a single unique rectangle that is a
    # single point. Increase the range by 1.0 to prevent divide by zero error
    for d in range(n):
        if dim_ranges[d][0] == dim_ranges[d][1]:
            dim_ranges[d] = (dim_ranges[d][0], dim_ranges[d][1] + 1)
    # Compute hilbert distance of the middle of each bounding box
    dim_mids = [(bounds[:, d] + bounds[:, d + n]) / 2.0 for d in range(n)]
    side_length = 2 ** p
    coords = np.zeros((bounds.shape[0], n), dtype=np.int64)
    for d in range(n):
        coords[:, d] = _data2coord(dim_mids[d], dim_ranges[d], side_length)
    hilbert_distances = distances_from_coordinates(p, coords)
    return hilbert_distances


class HilbertRtree:
    """
    This class provides a numba implementation of a read-only Hilbert R-tree
    spatial index

    See https://en.wikipedia.org/wiki/Hilbert_R-tree for more info on the Hilbert R-tree

    This implementation stores the R-tree as an array representation of a binary tree.
    See https://en.wikipedia.org/wiki/Binary_tree#Arrays for more info on the array
    representation of a binary tree.
    """

    @staticmethod
    @ngjit
    def _build_hilbert_rtree(bounds, p, page_size):
        """
        numba function to construct a Hilbert Rtree

        See HilbertRtree.__init__ for parameter descriptions
        """
        # Handle empty bounds array
        if bounds.size == 0:
            return bounds, np.zeros(0, dtype=np.int64), bounds

        # Init bounds_tree array for storing the binary tree representation
        input_size = bounds.shape[0]
        n = bounds.shape[1] // 2
        num_pages = int(np.ceil(input_size / page_size))

        tree_depth = int(np.ceil(np.log2(num_pages)))
        next_pow2 = 2 ** tree_depth
        tree_length = next_pow2 * 2 - 1
        bounds_tree = np.full((tree_length, 2 * n), np.nan)
        leaf_start = tree_length - next_pow2

        # Compute Hilbert distances for inputs
        total_bounds = ([bounds[:, d].min() for d in range(n)] +
                        [bounds[:, d + n].max() for d in range(n)])
        hilbert_distances = _distances_from_bounds(bounds, total_bounds, p)

        # Calculate indices needed to sort bounds by hilbert distance
        keys = np.argsort(hilbert_distances)

        # Populate leaves of the tree, one leaf per page. This is layer = tree_depth
        sorted_bounds = bounds[keys, :]

        for page in range(num_pages):
            start = page * page_size
            stop = start + page_size
            page_bounds = sorted_bounds[start:stop, :]
            d_mins = [np.min(page_bounds[:, d]) for d in range(n)]
            d_maxes = [np.max(page_bounds[:, d + n]) for d in range(n)]
            d_ranges = d_mins + d_maxes
            bounds_tree[leaf_start + page, :] = d_ranges

        # Populate internal layers of tree
        layer = tree_depth - 1
        start = _parent(tree_length - next_pow2)
        stop = _parent(tree_length - 1)

        while layer >= 0:
            for node in range(start, stop + 1):
                left_bounds = bounds_tree[_left_child(node), :]
                left_valid = not np.isnan(left_bounds[0])
                right_bounds = bounds_tree[_right_child(node), :]
                right_valid = not np.isnan(right_bounds[0])

                if left_valid:
                    if right_valid:
                        d_mins = [min(left_bounds[d], right_bounds[d]) for d in range(n)]
                        d_maxes = [max(left_bounds[d + n], right_bounds[d + n]) for d in range(n)]
                        d_ranges = d_mins + d_maxes
                        bounds_tree[node, :] = d_ranges
                    else:
                        bounds_tree[node, :] = left_bounds

                elif right_valid:
                    bounds_tree[node, :] = right_bounds

            # update layer, start/stop
            start = _parent(start)
            stop = _parent(stop)
            layer -= 1

        return sorted_bounds, keys, bounds_tree

    def __init__(self, bounds, p=10, page_size=512):
        """
        Construct a new HilbertRtree

        Args:
            bounds: A 2-dimensional numpy array representing a collection of
                n-dimensional bounding boxes. One row per bounding box with
                2*n columns containing the coordinates of each bounding box as follows:
                    - bounds[:, 0:n] contains the min values of the bounding boxes,
                        one column for each of the n dimensions
                    - bounds[:, n+1:2n] contains the max values of the bounding boxes,
                        one column for each of the n dimensions
            p: The Hilbert curve order parameter that determines the resolution
                of the 2D grid that data points are rounded to before computing
                their Hilbert distance. Points will be discretized into 2 ** p
                bins in both the x and y dimensions.
            page_size: Number of elements per leaf of the tree.
        """
        # Validate/coerce inputs
        if len(bounds.shape) != 2:
            raise ValueError("bounds must be a 2D array")

        if bounds.shape[1] < 2 or bounds.shape[1] % 2 != 0:
            raise ValueError(
                "The second dimension of bounds must be a multiple of 2 and at least 2"
            )

        self._page_size = max(1, page_size)  # 1 is smallest valid page size
        self._numba_rtree = None
        self._sorted_bounds, self._keys, self._bounds_tree = \
            HilbertRtree._build_hilbert_rtree(bounds.astype('float64'), p, self._page_size)

    def __getstate__(self):
        # Remove _NumbaRtree instance during serialization since jitclass instances
        # don't support it.
        state = self.__dict__
        state['_numba_rtree'] = None
        return state

    @property
    def numba_rtree(self):
        """
        Returns:
            _NumbaRtree jitclass instance that is suitable for use inside numba
            functions
        """
        if self._numba_rtree is None:
            self._numba_rtree = _NumbaRtree(
                self._sorted_bounds, self._keys, self._page_size, self._bounds_tree
            )
        return self._numba_rtree

    def intersects(self, bounds):
        """
        Compute the indices of the input bounding boxes that intersect with the
        supplied query bounds

        Args:
            bounds: An array of the form [min0, min1, ..., max0, max1, ...]
                representing the bounds to calculate intersections against

        Returns:
            1d numpy array of the indices of all of the rows in the input bounds array
            that intersect with the query bounds
        """
        bounds = tuple(float(b) for b in bounds)
        return self.numba_rtree.intersects(bounds)

    def covers_overlaps(self, bounds):
        """
        Simultaneously compute the indices of the input bounding boxes that are covered
        by the query bounds and those that overlap with the query bounds

        Args:
            bounds: An array of the form [min0, min1, ..., max0, max1, ...]
                representing the bounds to calculate intersections against

        Returns:
            Tuple of two 1d numpy arrays of indices into the input bounds array.
              * The first array contains indices of all bounding boxes that are fully
              covered by the query bounds.
              * The second array contains the indices of all bounding boxes that
              overlap with one or more edges of the query bounds.
        """
        bounds = tuple(float(b) for b in bounds)
        return self.numba_rtree.covers_overlaps(bounds)

    @property
    def empty(self):
        """
        True if the RTree was created with zero bounding boxes
        """
        return self.numba_rtree._bounds_tree.shape[0] == 0

    @property
    def total_bounds(self):
        """
        Tuple of the total bounds of all bounding boxes
        """
        if not self.empty:
            return tuple(self.numba_rtree._bounds_tree[0, :])
        else:
            return tuple((np.nan,) * self.numba_rtree._bounds_tree.shape[1])


_numbartree_spec = [
    ('_bounds', float64[:, :]),
    ('_keys', int64[:]),
    ('_page_size', int64),
    ('_bounds_tree', float64[:, :]),
]
@jitclass(_numbartree_spec)
class _NumbaRtree:
    def __init__(self, bounds, keys, page_size, bounds_tree):
        self._bounds = bounds
        self._keys = keys
        self._page_size = page_size
        self._bounds_tree = bounds_tree

    def _leaf_start(self):
        """
        Returns
            Index of the first leaf node in bounds_tree
        """
        return (self._bounds_tree.shape[0] + 1) // 2 - 1

    def _start_index(self, node):
        """
        Args
            node: Index into _bounds_tree representing a binary tree node
        Returns
            The first index into keys represented by node
        """
        leaf_start = self._leaf_start()
        while True:
            child = _left_child(node)
            if child >= self._bounds_tree.shape[0]:
                page = node - leaf_start
                return page * self._page_size
            else:
                node = child

    def _stop_index(self, node):
        """
        Args
            node: Index into _bounds_tree representing a binary tree node
        Returns
            One past the last index into keys represented by node
        """
        leaf_start = self._leaf_start()
        while True:
            child = _right_child(node)
            if child >= self._bounds_tree.shape[0]:
                page = node - leaf_start + 1
                return page * self._page_size
            else:
                node = child

    def _maybe_intersects_ranges(self, query_bounds):
        if len(query_bounds) % 2 != 0:
            raise ValueError(
                'query_bounds must an array with an even number of elements'
            )
        n = len(query_bounds) // 2
        nodes = [0]
        covered_ranges = []
        maybe_intersect_ranges = []
        # Find ranges of indices that overlap with query bounds
        while nodes:
            next_node = nodes.pop()
            node_bounds = self._bounds_tree[next_node, :]

            # Check if node's bounds are fully outside query bounds
            outside = False
            for d in range(n):
                if (query_bounds[n + d] < node_bounds[d] or
                        query_bounds[d] > node_bounds[n + d]):
                    outside = True
                    break

            if outside:
                continue

            # Check if node's bounds are fully inside query bounds
            inside = True
            for d in range(n):
                if (node_bounds[d] < query_bounds[d] or
                        node_bounds[n + d] > query_bounds[n + d]):
                    inside = False
                    break

            if inside:
                # Node's bounds are fully inside query bounds
                start = self._start_index(next_node)
                stop = self._stop_index(next_node)
                covered_ranges.append((start, stop))
            else:
                start = self._start_index(next_node)
                stop = self._stop_index(next_node)
                if stop - start <= self._page_size:
                    maybe_intersect_ranges.append((start, stop))
                else:
                    # Partial overlap of interior bounding box, recurse to children
                    nodes.extend([_right_child(next_node), _left_child(next_node)])

        return covered_ranges, maybe_intersect_ranges

    def intersects(self, query_bounds):
        """
        See HilbertRtree.intersection
        """
        if self._bounds.size == 0:
            return np.zeros(0, dtype=np.uint32)

        n = len(query_bounds) // 2
        covered_ranges, maybe_intersect_ranges = self._maybe_intersects_ranges(
            query_bounds)

        # Compute max result length
        max_len = 0
        for start, stop in covered_ranges:
            max_len += stop - start
        for start, stop in maybe_intersect_ranges:
            max_len += stop - start

        # Initialize result buffer with max length
        result = np.zeros(max_len, dtype=np.uint32)

        # populate result from covered_ranges. In this case, we have certainty that
        # all bounding boxes in each slice intersect with query_bounds.
        result_start = 0
        for start, stop in covered_ranges:
            next_slice = self._keys[start:stop]
            result[result_start:result_start + len(next_slice)] = next_slice
            result_start += len(next_slice)

        # populate result from maybe_intersect_ranges.  In this case, we need to use
        # a brute-force check to determine which bounding boxes in the slice intersect
        # with query_bounds and only include those.
        for start, stop in maybe_intersect_ranges:
            next_slice = self._keys[start:stop]
            bounds_slice = self._bounds[start:stop, :]
            outside_mask = np.zeros(bounds_slice.shape[0], dtype=np.bool_)
            for d in range(n):
                outside_mask |= (bounds_slice[:, d + n] < query_bounds[d])
                outside_mask |= (bounds_slice[:, d] > query_bounds[d + n])

            next_slice = next_slice[~outside_mask]
            result[result_start:result_start + len(next_slice)] = next_slice
            result_start += len(next_slice)

        # Return populated portion of result
        return result[:result_start]

    def covers_overlaps(self, query_bounds):
        """
        See HilbertRtree.covers_overlaps
        """
        if self._bounds.size == 0:
            return np.zeros(0, dtype=np.uint32), np.zeros(0, dtype=np.uint32)

        n = len(query_bounds) // 2
        covered_ranges, maybe_intersect_ranges = self._maybe_intersects_ranges(
            query_bounds)

        # Initialize results arrays
        max_len0 = 0
        for start, stop in covered_ranges:
            max_len0 += stop - start

        max_len1 = 0
        for start, stop in maybe_intersect_ranges:
            max_len1 += stop - start

        covers_inds = np.zeros(max_len0 + max_len1, dtype=np.uint32)
        overlaps_inds = np.zeros(max_len1, dtype=np.uint32)

        # Process covered_ranges, all of these will be added to covered_inds
        covers_start = 0
        for start, stop in covered_ranges:
            next_slice = self._keys[start:stop]
            covers_inds[covers_start:covers_start + len(next_slice)] = next_slice
            covers_start += len(next_slice)

        # Process maybe_intersect_ranges. Some of these will populate covered_inds,
        # some overlaps_inds, and some will be excluded
        overlaps_start = 0
        for start, stop in maybe_intersect_ranges:
            next_slice = self._keys[start:stop]
            bounds_slice = self._bounds[start:stop, :]

            # Check which bounds are fully outside query region
            outside_mask = np.zeros(bounds_slice.shape[0], dtype=np.bool_)
            for d in range(n):
                outside_mask |= (bounds_slice[:, d + n] < query_bounds[d])
                outside_mask |= (bounds_slice[:, d] > query_bounds[d + n])

            # Check which bounds are fully covered by query region
            covers_mask = np.ones(bounds_slice.shape[0], dtype=np.bool_)
            for d in range(n):
                covers_mask &= (bounds_slice[:, d] >= query_bounds[d])
                covers_mask &= (bounds_slice[:, d + n] <= query_bounds[d + n])

            # Update covers_inds
            covers_slice = next_slice[covers_mask]
            covers_inds[covers_start:covers_start + len(covers_slice)] = covers_slice
            covers_start += len(covers_slice)

            # Update overlaps_inds
            overlaps_slice = next_slice[~(outside_mask | covers_mask)]
            overlaps_stop = overlaps_start + len(overlaps_slice)
            overlaps_inds[overlaps_start:overlaps_stop] = overlaps_slice
            overlaps_start += len(overlaps_slice)

        return covers_inds[:covers_start], overlaps_inds[:overlaps_start]
