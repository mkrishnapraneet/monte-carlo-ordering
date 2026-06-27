import numpy as np
from scipy.stats import gaussian_kde
import heapq


def estimate_fes_2d(coords_2d, kT=0.596, grid_bins=200,
                    x_range=(-np.pi, np.pi), y_range=(-np.pi, np.pi),
                    periodic=False):
    """
    Estimate 2D free energy surface from MC ensemble coordinates.

    Parameters
    ----------
    coords_2d : np.ndarray, shape (n_frames, 2)
        2D coordinates (e.g. raw dihedral angles in radians, or Q+RMSD).
    kT : float
        Thermal energy in kcal/mol (default 0.596 for 300 K).
    grid_bins : int
        Number of grid points along each axis.
    x_range, y_range : tuple
        Range of each axis.
    periodic : bool
        If True, tile the data 3×3 with shifts of ±(x_period, y_period) before
        computing the KDE. This corrects for periodic boundary conditions (e.g.
        dihedral angles that wrap at ±π). The normalization F_norm = F - F.min()
        absorbs the 9× density factor from tiling. Default False.

    Returns
    -------
    F_norm : np.ndarray, shape (grid_bins, grid_bins)
        Free energy surface in kcal/mol, normalized so min = 0.
    x_grid : np.ndarray  (grid_bins,)
    y_grid : np.ndarray  (grid_bins,)
    """
    x_grid = np.linspace(x_range[0], x_range[1], grid_bins)
    y_grid = np.linspace(y_range[0], y_range[1], grid_bins)
    XX, YY = np.meshgrid(x_grid, y_grid, indexing='ij')   # shape (G, G)
    grid_points = np.vstack([XX.ravel(), YY.ravel()])      # shape (2, G*G)

    if periodic:
        # Tile the data 3×3 to handle wrap-around at the boundaries.
        # Frames near +π and -π represent the same physical state; without tiling
        # the KDE splits their density across opposite edges, inflating the FES there.
        period_x = x_range[1] - x_range[0]
        period_y = y_range[1] - y_range[0]
        tiles = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                tiles.append(coords_2d + np.array([dx * period_x, dy * period_y]))
        coords_for_kde = np.vstack(tiles)
    else:
        coords_for_kde = coords_2d

    if periodic:
        # 3×3 tiling inflates std by ~2.17× while reducing Scott's factor by 9^(-1/6)≈0.69,
        # so the effective bandwidth is ~1.5× too wide. Correct by computing Scott's factor
        # from the original n and scaling it for the tiled std.
        n_orig = len(coords_2d)
        d = coords_2d.shape[1]
        h_orig = n_orig ** (-1.0 / (d + 4))
        std_orig = np.mean(coords_2d.std(axis=0))
        std_tiled = np.mean(coords_for_kde.std(axis=0))
        bw_corrected = h_orig * (std_orig / std_tiled)
        kde = gaussian_kde(coords_for_kde.T, bw_method=bw_corrected)
    else:
        kde = gaussian_kde(coords_2d.T, bw_method='scott')
    density = kde(grid_points).reshape(grid_bins, grid_bins)
    density = np.clip(density, 1e-300, None)

    F = -kT * np.log(density)
    F_norm = F - F.min()

    return F_norm, x_grid, y_grid


def coord_to_grid_idx(coord_val, axis_range, grid_bins):
    """Convert a single coordinate value to its nearest grid index."""
    lo, hi = axis_range
    idx = int(round((coord_val - lo) / (hi - lo) * (grid_bins - 1)))
    return np.clip(idx, 0, grid_bins - 1)


def find_saddle_energy(F_norm, start_idx, end_idx, periodic=False):
    """
    Find the saddle-point free energy between two grid points.
    Uses modified Dijkstra (minimax path = lowest mountain pass).

    Parameters
    ----------
    F_norm : np.ndarray, shape (G, G)
        Normalised free energy grid (min = 0).
    start_idx : tuple (i, j)
        Grid index of the starting centroid.
    end_idx : tuple (i, j)
        Grid index of the ending centroid.
    periodic : bool
        If True, wrap neighbor coordinates with modulo G (toroidal grid).
        Required when the two centroids are near opposite edges of the domain
        (e.g. cluster 2 at chi≈+π and cluster 5 at chi≈−π). Default False.

    Returns
    -------
    saddle_F : float
        Free energy at the lowest saddle point (kcal/mol).
    """
    G = F_norm.shape[0]
    heap = [(F_norm[start_idx], start_idx)]
    visited = {}

    neighbours = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

    while heap:
        cost, pos = heapq.heappop(heap)
        if pos in visited:
            continue
        visited[pos] = cost

        if pos == end_idx:
            return cost

        r, c = pos
        for dr, dc in neighbours:
            if periodic:
                nr, nc = (r + dr) % G, (c + dc) % G
            else:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < G and 0 <= nc < G):
                    continue
            if (nr, nc) not in visited:
                new_cost = max(cost, F_norm[nr, nc])
                heapq.heappush(heap, (new_cost, (nr, nc)))

    return float('inf')


def compute_direct_saddles(F_norm, seed_indices, periodic=False, connectivity=8):
    """
    Direct saddle free energy between every pair of basins, via a priority-flood
    watershed of the FES.

    Floods cells outward from the seed (centroid) cells in order of increasing
    free energy, growing one basin per seed. When two basins' territories first
    meet, the flood level there — max(F) of the two adjoining cells — is the
    lowest point on the ridge that *directly* separates them, i.e. the barrier of
    a path that does not pass through any third basin. This is the correct
    activation barrier for a single-step transition.

    It replaces the unconstrained minimax (`find_saddle_energy`), whose lowest
    path is free to dip down through an intervening deeper basin (e.g. two gauche
    wells separated by the trans well): that yields an *ultrametric* barrier — the
    same "exit the starting basin" value for every destination — and cannot
    distinguish a direct transition from one mediated by a third state.

    Parameters
    ----------
    F_norm : np.ndarray, shape (G, G)
        Normalised free energy grid (min = 0).
    seed_indices : list of (i, j) tuples, length N
        Grid index of each basin's centroid (one seed per state).
    periodic : bool
        If True, wrap neighbour coordinates with modulo G (toroidal grid).
    connectivity : int
        4 or 8 neighbour stencil (default 8, matching `find_saddle_energy`).

    Returns
    -------
    saddles : np.ndarray, shape (N, N)
        Direct saddle free energy for each basin pair (symmetric, np.inf on the
        diagonal and for pairs whose territories never share a border — i.e. no
        direct single-pass path exists between them).
    """
    G = F_norm.shape[0]
    N = len(seed_indices)

    label = -np.ones((G, G), dtype=int)
    heap = []
    for b, (r, c) in enumerate(seed_indices):
        label[r, c] = b
        heapq.heappush(heap, (F_norm[r, c], r, c))

    if connectivity == 4:
        neighbours = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        neighbours = [(-1, 0), (1, 0), (0, -1), (0, 1),
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]

    saddles = np.full((N, N), np.inf)

    while heap:
        _, r, c = heapq.heappop(heap)
        lb = label[r, c]
        for dr, dc in neighbours:
            if periodic:
                nr, nc = (r + dr) % G, (c + dc) % G
            else:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < G and 0 <= nc < G):
                    continue
            nlb = label[nr, nc]
            if nlb == -1:
                label[nr, nc] = lb
                heapq.heappush(heap, (F_norm[nr, nc], nr, nc))
            elif nlb != lb:
                # two basins meet here -> ridge between them; keep lowest pass
                lvl = max(F_norm[r, c], F_norm[nr, nc])
                if lvl < saddles[lb, nlb]:
                    saddles[lb, nlb] = lvl
                    saddles[nlb, lb] = lvl

    return saddles
