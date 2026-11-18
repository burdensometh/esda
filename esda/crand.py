"""
Centralised conditional randomisation engine. Numba accelerated.
"""

import importlib
import os
import warnings

import numpy as np

from .permutation_plan import PermutationPlan
from .significance import _permutation_significance

try:
    from numba import boolean, njit, prange
except (ImportError, ModuleNotFoundError):
    from libpysal.common import jit as njit

    prange = range
    boolean = bool


__all__ = ["crand"]

#######################################################################
#                   Utilities for all functions                       #
#######################################################################


@njit(fastmath=True)
def vec_permutations(max_card: int, n: int, k_replications: int, seed: int):
    """
    Generate `max_card` permuted IDs, sampled from `n` without replacement,
    `k_replications` times
    ...

    Parameters
    ----------
    max_card : int
        Number of permuted IDs to generate per sample
    n : int
        Size of the sample to sample IDs from
    k_replications : int
        Number of samples of permuted IDs to perform
    seed : int
        Seed to ensure reproducibility of conditional randomizations

    Returns
    -------
    result : ndarray
        (k_replications, max_card) array with permuted IDs
    """
    np.random.seed(seed)
    result = np.empty((k_replications, max_card), dtype=np.int64)
    for k in prange(k_replications):
        result[k] = np.random.choice(n - 1, size=max_card, replace=False)
    return result


def crand(
    z,
    w,
    observed,
    permutations,
    keep,
    n_jobs,
    stat_func,
    scaling=None,
    seed=None,
    island_weight=0,
    alternative=None,
    permutation_plan=None,
):
    """
    Conduct conditional randomization of a given input using the provided
    statistic function. Numba accelerated.
    ...

    Parameters
    ----------
    z : ndarray
        2D array with N rows with standardised observed values
    w : libpysal.weights.W
        Spatial weights object
    observed : ndarray
        (N,) array with observed values
    permutations : int
        Number of permutations for conditional randomisation
    keep : Boolean
        If True, store simulation; else do not return randomised statistics
    n_jobs : int
        Number of cores to be used in the conditional randomisation. If -1,
        all available cores are used.
    stat_func : callable
        Method implementing the spatial statistic to be evaluated under
        conditional randomisation. The method needs to have the following
        signature:
            i : int
                Position of observation to be evaluated in the sample
            z : ndarray
                2D array with N rows with standardised observed values
            permuted_ids : ndarray
                (permutations, max_cardinality) array with indices of permuted
                IDs
            weights_i : ndarray
                Weights for neighbors in i
            scaling : float
                Scaling value to apply to every local statistic
     seed : None/int
        Seed to ensure reproducibility of conditional randomizations
    alternative : None | str = None
        The alternative hypothesis for conditional randomization. The current behavior
        is ``None``, which defaults to ``'directed'``. We strongly recommend moving
        to ``'two-sided'``. Valid alternatives include:
            * 'two-sided'
            * 'greater'
            * 'lesser'
            * 'directed' -- Current default behavior
            * 'folded'
    permutation_plan : PermutationPlan, optional
        A shared, stable-region-ID-bound permutation plan. When provided, the
        same on-demand permutation blocks drive this call and every other
        statistic consuming the plan, regardless of row order or serial /
        parallel chunking. The plan is validated against ``w`` and ``z``
        before any work and incompatible inputs are rejected. ``seed`` must
        then be None and ``permutations`` must match the plan. Blocks are
        streamed, so the full simulation cube is never materialised unless
        ``keep`` is True.

    Returns
    -------
    p_sim : ndarray
        (N,) array with pseudo p-values from conditional permutation
    rlocals : ndarray
        If keep=True, (N, permutations) array with simulated values
        of stat_func under the null of spatial randomness; else, empty (1, 1) array
    """
    adj_matrix = w.sparse

    n = len(z)
    if z.ndim == 2:
        if z.shape[1] == 2:
            # assume that matrix is [X Y], and scaling is moran-like
            scaling = (
                (n - 1) / (z[:, 0] * z[:, 0]).sum() if (scaling is None) else scaling
            )
        elif z.shape[1] == 1:
            # assume that matrix is [X], and scaling is moran-like
            scaling = (n - 1) / (z * z).sum() if (scaling is None) else scaling
        else:
            raise NotImplementedError(
                f"multivariable input is not yet supported in "
                f"conditional randomization. Received `z` of shape {z.shape}"
            )
    elif z.ndim == 1:
        scaling = (n - 1) / (z * z).sum() if (scaling is None) else scaling
    else:
        raise NotImplementedError(
            f"multivariable input is not yet supported in "
            f"conditional randomization. Received `z` of shape {z.shape}"
        )

    if alternative is None:
        warnings.warn(
            "The alternative hypothesis for conditional randomization"
            " is changing in the next major release of esda. We recommend"
            " setting alternative='two-sided', which will generally"
            " double the p-value returned."
            " To retain the current behavior, set alternative='directed'."
            " We strongly recommend moving to alternative='two-sided'.",
            DeprecationWarning,
            stacklevel=2,
        )
        # TODO: replace this with 'two-sided' by next major release
        alternative = "directed"
    if alternative not in ("two-sided", "greater", "lesser", "directed", "folded"):
        raise ValueError(
            f"alternative='{alternative}' provided, but is not"
            " one of the supported options: 'two-sided', 'greater', "
            "'lesser', 'directed', 'folded')"
        )

    if permutation_plan is not None:
        if seed is not None:
            raise ValueError(
                "Pass either `permutation_plan` or `seed`, not both: the plan "
                "already records the random seed and mixing the two would "
                "break reproducibility across shared statistics."
            )
        return _crand_with_plan(
            z=z,
            w=w,
            observed=observed,
            keep=keep,
            n_jobs=n_jobs,
            stat_func=stat_func,
            scaling=scaling,
            island_weight=island_weight,
            alternative=alternative,
            plan=permutation_plan,
            permutations=permutations,
        )

    # paralellise over permutations?
    if seed is None:
        seed = np.random.randint(12345, 12345000)

    # we need to be careful to shuffle only *other* sites, not
    # the self-site. This means we need to
    # extract the self-weight, if any
    self_weights = adj_matrix.diagonal()
    # force the self-site weight to zero
    with warnings.catch_warnings():
        # massive changes to sparsity incur a cost, but it's not
        # large for simply changing the diag
        warnings.simplefilter("ignore")
        adj_matrix.setdiag(0)
        adj_matrix.eliminate_zeros()
    # extract the weights from a now no-self-weighted adj_matrix
    other_weights = adj_matrix.data.astype(z.dtype)  # cast is forced by @ in numba
    # use the non-self weight as the cardinality, since
    # this is the set we have to randomize.
    # if there is a self-neighbor, we need to *not* shuffle the
    # self neighbor, since conditional randomization conditions on site i.
    cardinalities = np.array((adj_matrix != 0).sum(1)).flatten()
    max_card = cardinalities.max()
    permuted_ids = vec_permutations(max_card, n, permutations, seed)

    if n_jobs != 1 and not importlib.util.find_spec("joblib"):
        warnings.warn(
            f"Parallel processing is requested (n_jobs={n_jobs}),"
            f" but joblib cannot be imported. n_jobs will be set"
            f" to 1.",
            stacklevel=2,
        )
        n_jobs = 1

    if n_jobs == 1:
        p_sims, rlocals = compute_chunk(
            0,  # chunk start
            z,  # chunked z, for serial this is the entire data
            z,  # all z, for serial this is also the entire data
            observed,  # observed statistics
            cardinalities,  # cardinalities conforming to chunked z
            self_weights,  # n-length vector containing the self-weights.
            other_weights,  # flat weights buffer
            permuted_ids,  # permuted ids
            scaling,  # scaling applied to all statistics
            keep,  # whether or not to keep the local statistics
            stat_func,
            island_weight,
            alternative=alternative,
        )
    else:
        if n_jobs == -1:
            n_jobs = os.cpu_count()
        if n_jobs > len(z):
            n_jobs = len(z)
        # Parallel implementation
        p_sims, rlocals = parallel_crand(
            z,
            observed,
            cardinalities,
            self_weights,
            other_weights,
            permuted_ids,
            scaling,
            n_jobs,
            keep,
            stat_func,
            island_weight,
            alternative=alternative,
        )

    return p_sims, rlocals


@njit(parallel=False, fastmath=True)
def compute_chunk(
    chunk_start: int,
    z_chunk: np.ndarray,
    z: np.ndarray,
    observed: np.ndarray,
    cardinalities: np.ndarray,
    self_weights: np.ndarray,
    other_weights: np.ndarray,
    permuted_ids: np.ndarray,
    scaling: np.float64,
    keep: bool,
    stat_func,
    island_weight: float,
    alternative: str,
):
    """
    Compute conditional randomisation for a single chunk
    ...

    Parameters
    ----------
    chunk_start : int
        Starting index for the chunk of input. Should be zero if z_chunk == z.
    z_chunk : numpy.ndarray
        (n_chunk,) array containing the chunk of standardised observed values.
    z : ndarray
        2D array with N rows with standardised observed values
    observed : ndarray
        (n_chunk,) array containing observed values for the chunk
    cardinalities : ndarray
        (n_chunk,) array containing the cardinalities for each element.
    self_weights : ndarray of shape (n,)
        Array containing the self-weights for each observation. In most cases, this
        will be zero. But, in some cases (e.g. Gi-star or kernel weights), this will
        be nonzero.
    other_weights : ndarray
        Array containing the weights of all other sites in the computation
        other than site i. If self_weights is zero, this has as many entries
        as the sum of `cardinalities`.
    permuted_ids : ndarray
        (permutations, max_cardinality) array with indices of permuted
        ids to use to construct random realizations of the statistic
    scaling : float
        Scaling value to apply to every local statistic
    keep : bool
        If True, store simulation; else do not return randomised statistics
    stat_func : callable
        Method implementing the spatial statistic to be evaluated under
        conditional randomisation. The method needs to have the following
        signature:
            i : int
                Position of observation to be evaluated in the sample
            z : ndarray
                2D array with N rows with standardised observed values
            permuted_ids : ndarray
                (permutations, max_cardinality) array with indices of permuted
                IDs
            weights_i : ndarray
                Weights for neighbors in i
            scaling : float
                Scaling value to apply to every local statistic
    island_weight:
        value to use as a weight for the "fake" neighbor for every island.
        If numpy.nan, will propagate to the final local statistic depending
        on the `stat_func`. If 0, then the lag is always zero for islands.

    Returns
    -------
    larger : ndarray
        (n_chunk,) array with number of random draws under the null larger
        than observed value of statistic
    rlocals : ndarray
        (n_chunk, max_cardinality) array with local statistics simulated under
        the null of spatial randomness
    """
    chunk_n = z_chunk.shape[0]
    n_samples = z.shape[0]
    p_permutations, k_max_card = permuted_ids.shape
    p_sims = np.zeros((chunk_n,), dtype=np.float32)
    rlocals = np.empty((chunk_n, permuted_ids.shape[0])) if keep else np.empty((1, 1))

    mask = np.ones((n_samples,), dtype=np.int8) == 1
    wloc = 0

    for i in range(chunk_n):
        cardinality = cardinalities[i]
        if cardinality == 0:  # deal with islands
            weights_i = np.zeros(2, dtype=other_weights.dtype)
            weights_i[1] = island_weight
        else:
            # we need to fix the self-weight to the first position
            weights_i = np.zeros(cardinality + 1, dtype=other_weights.dtype)
            weights_i[0] = self_weights[i]
            # this chomps the next `cardinality` weights off of `weights`
            weights_i[1:] = other_weights[wloc : (wloc + cardinality)]
        wloc += cardinality
        mask[chunk_start + i] = False
        rstats = stat_func(chunk_start + i, z, permuted_ids, weights_i, scaling)
        p_sims[i] = _permutation_significance(
            observed[i], rstats, alternative=alternative
        ).item()
        if keep:
            rlocals[i] = rstats

    return p_sims, rlocals


#######################################################################
#                   Parallel Implementation                           #
#######################################################################


@njit(fastmath=True)
def build_weights_offsets(cardinalities: np.ndarray, n_chunks: int):
    """
    Utility function to construct offsets into the weights
    flat data array found in the W.sparse.data object
    ...

    Parameters
    ----------
    cardinalities : ndarray
        (n_chunk,) array containing the cardinalities for each element.
    n_chunks : int
        Number of chunks to split the weights into

    Returns
    -------
    boundary_points : ndarray
        (n_chunks,) array with positions to split a flat representation of W
        for every chunk
    """
    boundary_points = np.zeros((n_chunks + 1,), dtype=np.int64)
    n = cardinalities.shape[0]
    chunk_size = np.int64(n / n_chunks) + 1
    start = 0
    for i in range(n_chunks):
        advance = cardinalities[start : start + chunk_size].sum()
        boundary_points[i + 1] = boundary_points[i] + advance
        start += chunk_size
    return boundary_points


@njit(fastmath=True)
def chunk_generator(
    n_jobs: int,
    starts: np.ndarray,
    z: np.ndarray,
    observed: np.ndarray,
    cardinalities: np.ndarray,
    self_weights: np.ndarray,
    other_weights: np.ndarray,
    w_boundary_points: np.ndarray,
):
    """
    Construct chunks to iterate over within numba in parallel
    ...

    Parameters
    ----------
    n_jobs : int
        Number of cores to be used in the conditional randomisation. If -1,
        all available cores are used.
    starts : ndarray
        (n_chunks+1,) array of positional starts for each chunk
    z : ndarray
        2D array with N rows with standardised observed values
    observed : ndarray
        (N,) array with observed values
    cardinalities : ndarray
        (N,) array containing the cardinalities for each element.
    weights : ndarray
        Array containing the weights within the chunk in a flat format (ie. as
        obtained from the `values` attribute of a CSR sparse representation of
        the original W. This is as long as the sum of `cardinalities`
    w_boundary_points : ndarray
        (n_chunks,) array with positions to split a flat representation of W
        for every chunk

    Yields
    ------
    start : int
        Starting index for the chunk of input. Should be zero if z_chunk == z.
    z_chunk : numpy.ndarray
        (n_chunk,) array containing the chunk of standardised observed values.
    z : ndarray
        2D array with N rows with standardised observed values
    observed_chunk : ndarray
        (n_chunk,) array containing observed values for the chunk
    cardinalities_chunk : ndarray
        (n_chunk,) array containing the cardinalities for each element.
    weights_chunk : ndarray
        Array containing the weights within the chunk in a flat format (ie. as
        obtained from the `values` attribute of a CSR sparse representation of
        the original W. This is as long as the sum of `cardinalities`
    """
    chunk_size = starts[1] - starts[0]
    for i in range(n_jobs):
        start = starts[i]
        z_chunk = z[start : (start + chunk_size)]
        self_weights_chunk = self_weights[start : (start + chunk_size)]
        observed_chunk = observed[start : (start + chunk_size)]
        cardinalities_chunk = cardinalities[start : (start + chunk_size)]
        w_chunk = other_weights[w_boundary_points[i] : w_boundary_points[i + 1]]
        yield (
            start,
            z_chunk,
            z,
            observed_chunk,
            cardinalities_chunk,
            self_weights_chunk,
            w_chunk,
        )


def parallel_crand(
    z: np.ndarray,
    observed: np.ndarray,
    cardinalities: np.ndarray,
    self_weights: np.ndarray,
    other_weights: np.ndarray,
    permuted_ids: np.ndarray,
    scaling: np.float64,
    n_jobs: int,
    keep: bool,
    stat_func,
    island_weight,
    alternative: str = "directed",
):
    """
    Conduct conditional randomization in parallel using numba
    ...

    Parameters
    ----------
    z : ndarray
        2D array with N rows with standardised observed values
    observed : ndarray
        (N,) array with observed values
    cardinalities : ndarray
        (N,) array containing the cardinalities for each element.
    self_weights : ndarray of shape (n,)
        Array containing the self-weights for each observation. In most cases, this
        will be zero. But, in some cases (e.g. Gi-star or kernel weights), this will
        be nonzero.
    other_weights : ndarray
        Array containing the weights of all other sites in the computation
        other than site i. If self_weights is zero, this has as many entries
        as the sum of `cardinalities`.
    permuted_ids : ndarray
        (permutations, max_cardinality) array with indices of permuted
        ids to use to construct random realizations of the statistic
    scaling : float64
        Scaling value to apply to every local statistic
    n_jobs : int
        Number of cores to be used in the conditional randomisation. If -1,
        all available cores are used.
    keep : Boolean
        If True, store simulation; else do not return randomised statistics
    stat_func : callable
        Method implementing the spatial statistic to be evaluated under
        conditional randomisation. The method needs to have the following
        signature:
            i : int
                Position of observation to be evaluated in the sample
            z : ndarray
                2D array with N rows with standardised observed values
            permuted_ids : ndarray
                (permutations, max_cardinality) array with indices of permuted
                IDs
            weights_i : ndarray
                Weights for neighbors in i
            scaling : float
                Scaling value to apply to every local statistic
    island_weight:
        value to use as a weight for the "fake" neighbor for every island.
        If numpy.nan, will propagate to the final local statistic depending
        on the `stat_func`. If 0, then the lag is always zero for islands.
    Returns
    -------
    larger : ndarray
        (N,) array with number of random draws under the null larger
        than observed value of statistic
    rlocals : ndarray
        (N, max_cardinality) array with local statistics simulated under
        the null of spatial randomness
    """
    from joblib import Parallel, delayed, parallel_backend

    n = z.shape[0]
    w_boundary_points = build_weights_offsets(cardinalities, n_jobs)
    chunk_size = n // n_jobs + 1
    starts = np.arange(n_jobs + 1) * chunk_size
    # ------------------------------------------------------------------
    # Set up output holders
    rlocals = np.empty((n, permuted_ids.shape[0])) if keep else np.empty((1, 1))

    # ------------------------------------------------------------------
    # Joblib parallel loop by chunks

    # construct chunks using a generator
    chunks = chunk_generator(
        n_jobs,
        starts,
        z,
        observed,
        cardinalities,
        self_weights,
        other_weights,
        w_boundary_points,
    )

    with parallel_backend("loky", inner_max_num_threads=1):
        worker_out = Parallel(n_jobs=n_jobs)(
            delayed(compute_chunk)(
                *pars,
                permuted_ids,
                scaling,
                keep,
                stat_func,
                island_weight,
                alternative,
            )
            for pars in chunks
        )

    p_sims, rlocals = zip(*worker_out, strict=True)
    p_sims = np.hstack(p_sims).squeeze()
    rlocals = np.vstack(rlocals).squeeze()
    return p_sims, rlocals


#######################################################################
#          Shared permutation-plan implementation (streamed)           #
#######################################################################


def _resolve_scaling(z, scaling):
    """Replicate the default moran-like scaling used by ``crand``."""
    n = len(z)
    if scaling is not None:
        return scaling
    if z.ndim == 2:
        if z.shape[1] == 2:
            return (n - 1) / (z[:, 0] * z[:, 0]).sum()
        elif z.shape[1] == 1:
            return (n - 1) / (z * z).sum()
        else:
            raise NotImplementedError(
                f"multivariable input is not yet supported in "
                f"conditional randomization. Received `z` of shape {z.shape}"
            )
    elif z.ndim == 1:
        return (n - 1) / (z * z).sum()
    else:
        raise NotImplementedError(
            f"multivariable input is not yet supported in "
            f"conditional randomization. Received `z` of shape {z.shape}"
        )


def _crand_with_plan(
    z,
    w,
    observed,
    keep,
    n_jobs,
    stat_func,
    scaling,
    island_weight,
    alternative,
    plan,
    permutations,
):
    """
    Run conditional randomisation consuming a shared :class:`PermutationPlan`.

    Inputs are reordered into the plan's canonical region-ID order before
    computation and outputs are mapped back to the caller's order, so a
    permutation attached to a stable region ID is the same draw regardless
    of the input row order. Permutation blocks are generated on demand and
    reduced into tail counters (or an optional simulation matrix), so the
    full ``(n, permutations)`` cube is never held when ``keep`` is False.
    """
    if not isinstance(plan, PermutationPlan):
        raise TypeError(
            "`permutation_plan` must be a PermutationPlan instance; got "
            f"{type(plan)!r}."
        )
    if permutations != plan.permutations:
        raise ValueError(
            f"PermutationPlan covers {plan.permutations} permutations but "
            f"{permutations} were requested. Reuse the plan as built or "
            "rebuild it with the matching count."
        )
    # Reject incompatible weights / candidate population before any draw.
    plan.check_compatible(w, z)

    z = np.asarray(z)
    observed = np.asarray(observed, dtype=np.float64).flatten()
    scaling = _resolve_scaling(z, scaling)

    # Reorder every input into the plan's canonical order, which is defined
    # by the sorted stable region identifiers. This makes a draw attached to
    # a stable ID identical across callers that supplied different row
    # orders; ``inverse_order`` maps canonical results back to caller order.
    id_order = (
        w.id_order if hasattr(w, "id_order") else w.unique_ids
    )
    canonical_order = plan.canonical_positions(id_order)
    inverse_order = np.argsort(canonical_order)

    z_canonical = z[canonical_order]
    observed_canonical = observed[canonical_order]

    adj_matrix = w.sparse[canonical_order, :][:, canonical_order]
    self_weights = adj_matrix.diagonal()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adj_matrix.setdiag(0)
        adj_matrix.eliminate_zeros()
    other_weights = adj_matrix.data.astype(z_canonical.dtype)
    cardinalities = np.array((adj_matrix != 0).sum(1)).flatten()

    if plan.max_cardinality != int(cardinalities.max(initial=0)):
        raise ValueError(
            "PermutationPlan max_cardinality "
            f"({plan.max_cardinality}) does not match the maximum neighbor "
            f"cardinality of the provided weights ({int(cardinalities.max())})."
        )

    if n_jobs != 1 and not importlib.util.find_spec("joblib"):
        warnings.warn(
            f"Parallel processing is requested (n_jobs={n_jobs}),"
            f" but joblib cannot be imported. n_jobs will be set"
            f" to 1.",
            stacklevel=2,
        )
        n_jobs = 1

    common = dict(
        z=z_canonical,
        observed=observed_canonical,
        cardinalities=cardinalities,
        self_weights=self_weights,
        other_weights=other_weights,
        scaling=scaling,
        stat_func=stat_func,
        island_weight=island_weight,
        alternative=alternative,
        plan=plan,
        # After canonical reordering, local position i already coincides with
        # canonical position i, so the focal mapping is the identity.
        focal_canonical=np.arange(len(z_canonical), dtype=np.int64),
    )

    if n_jobs == 1:
        p_sims, rlocals = _stream_plan_crand_serial(keep=keep, **common)
    else:
        if n_jobs == -1:
            n_jobs = os.cpu_count()
        n_jobs = min(n_jobs, len(z_canonical), plan.n_blocks)
        n_jobs = max(n_jobs, 1)
        p_sims, rlocals = _stream_plan_crand_parallel(
            keep=keep, n_jobs=n_jobs, **common
        )

    # Map results back to the caller's original region order.
    p_sims = np.asarray(p_sims, dtype=np.float32)[inverse_order]
    if keep:
        rlocals = np.asarray(rlocals)[inverse_order, :]
    return p_sims, rlocals


def _empty_tail_state(n, permutations, keep):
    """Allocate streaming tail counters and the optional simulation cube."""
    greater = np.zeros(n, dtype=np.int64)
    lesser = np.zeros(n, dtype=np.int64)
    rlocals = np.empty((n, permutations), dtype=np.float64) if keep else None
    return greater, lesser, rlocals


@njit(fastmath=False)
def _remap_rows_for_focal(block, focal_canonical_i, out):
    """
    Remap a stable-canonical-position block into the excluded-position
    space of one focal site, in place into ``out``.

    Plan blocks index the full candidate population in canonical (stable-ID)
    order, so a draw denotes the same region regardless of the caller's row
    order. The kernel receives every input pre-reordered into that canonical
    order, hence local and canonical positions coincide and the statistic's
    masked array simply omits the focal row. Canonical position ``p`` then
    maps to ``p`` when ``p < focal_canonical_i`` and to ``p - 1`` when
    ``p > focal_canonical_i``; the focal position itself never appears
    because the focal region is excluded from its own candidate set.
    """
    p_permutations, max_cardinality = block.shape
    for k in range(p_permutations):
        for j in range(max_cardinality):
            p = block[k, j]
            out[k, j] = p - 1 if p > focal_canonical_i else p
    return out


@njit(parallel=False, fastmath=True)
def compute_chunk_plan(
    z,
    observed,
    cardinalities,
    self_weights,
    other_weights,
    block,
    scaling,
    keep,
    stat_func,
    island_weight,
    alternative,
    focal_canonical,
):
    """
    Compute one on-demand plan block for every focal site.

    Unlike :func:`compute_chunk`, the block carries stable canonical
    positions and is remapped per focal site before invoking the statistic,
    which is what makes draws invariant to analysis-order reordering.
    ``focal_canonical[i]`` is the canonical position of the region at local
    position ``i``; it equals ``i`` because inputs are pre-reordered into
    the plan's canonical order, but is passed explicitly so the remap stays
    independent of any downstream chunking.

    Tail counters are returned instead of a block-local p-value, so blocks
    can be streamed without retaining the full simulation cube.
    """
    n_samples = z.shape[0]
    p_permutations, max_cardinality = block.shape
    greater = np.zeros((n_samples,), dtype=np.int64)
    lesser = np.zeros((n_samples,), dtype=np.int64)
    rlocals = (
        np.empty((n_samples, p_permutations), dtype=np.float64)
        if keep
        else np.empty((1, 1))
    )
    remapped = np.empty_like(block)
    wloc = 0

    for i in range(n_samples):
        cardinality = cardinalities[i]
        if cardinality == 0:  # deal with islands
            weights_i = np.zeros(2, dtype=other_weights.dtype)
            weights_i[1] = island_weight
        else:
            weights_i = np.zeros(cardinality + 1, dtype=other_weights.dtype)
            weights_i[0] = self_weights[i]
            weights_i[1:] = other_weights[wloc : (wloc + cardinality)]
        wloc += cardinality
        _remap_rows_for_focal(block, focal_canonical[i], remapped)
        rstats = stat_func(i, z, remapped, weights_i, scaling)
        if alternative == "greater" or alternative == "two-sided" or (
            alternative == "directed"
        ):
            greater[i] = np.sum(rstats >= observed[i])
        if alternative == "lesser" or alternative == "two-sided" or (
            alternative == "directed"
        ):
            lesser[i] = np.sum(rstats <= observed[i])
        if keep:
            rlocals[i] = rstats

    return greater, lesser, rlocals


def _finalize_pvalues(
    greater, lesser, observed, rlocals, alternative, permutations
):
    """Turn streamed counters into pseudo p-values using the existing rules."""
    denom = permutations + 1.0
    if alternative == "directed":
        larger = greater.copy()
        low_extreme = (permutations - larger) < larger
        larger[low_extreme] = permutations - larger[low_extreme]
        p_value = (larger + 1.0) / denom
    elif alternative == "lesser":
        p_value = (lesser + 1.0) / denom
    elif alternative == "greater":
        p_value = (greater + 1.0) / denom
    elif alternative == "two-sided":
        p_value = np.minimum(
            2.0 * (np.minimum(greater, lesser) + 1.0) / denom, 1.0
        )
    elif alternative == "folded":
        # Same folded tail rule as _permutation_significance: fold both
        # observed and simulated values about the simulated mean, then count
        # simulated folds at least as extreme as the observed fold. Computed
        # in vectorised form to match the streaming dtypes/shapes.
        means = rlocals.mean(axis=1, keepdims=True)
        folded_test_stat = np.abs(observed[:, None] - means)
        folded_reference = np.abs(rlocals - means)
        p_value = (
            (folded_reference >= folded_test_stat).sum(axis=1) + 1.0
        ) / denom
    else:  # pragma: no cover - validated upstream in crand()
        raise ValueError(f"unsupported alternative {alternative!r}")
    return p_value.astype(np.float32)


def _stream_plan_crand_serial(
    z,
    observed,
    cardinalities,
    self_weights,
    other_weights,
    scaling,
    stat_func,
    island_weight,
    alternative,
    plan,
    keep,
    focal_canonical,
):
    """Serial sweep over on-demand plan blocks with streaming reduction."""
    n = z.shape[0]
    greater, lesser, rlocals = _empty_tail_state(
        n, plan.permutations, keep
    )
    for start, block in plan.iter_blocks():
        stop = start + block.shape[0]
        b_greater, b_lesser, b_rlocals = compute_chunk_plan(
            z,
            observed,
            cardinalities,
            self_weights,
            other_weights,
            block,
            scaling,
            keep,
            stat_func,
            island_weight,
            alternative,
            focal_canonical,
        )
        greater += b_greater
        lesser += b_lesser
        if keep:
            rlocals[:, start:stop] = b_rlocals
    p_sims = _finalize_pvalues(
        greater, lesser, observed, rlocals, alternative, plan.permutations
    )
    return p_sims, rlocals if keep else np.empty((1, 1))


def _plan_block_worker(
    block_index,
    start,
    z,
    observed,
    cardinalities,
    self_weights,
    other_weights,
    scaling,
    stat_func,
    island_weight,
    alternative,
    plan,
    keep,
    focal_canonical,
):
    """Compute one plan block; module-level so joblib workers can pickle it."""
    block = plan.get_block(block_index)
    b_greater, b_lesser, b_rlocals = compute_chunk_plan(
        z,
        observed,
        cardinalities,
        self_weights,
        other_weights,
        block,
        scaling,
        keep,
        stat_func,
        island_weight,
        alternative,
        focal_canonical,
    )
    return start, b_greater, b_lesser, b_rlocals


def _stream_plan_crand_parallel(
    z,
    observed,
    cardinalities,
    self_weights,
    other_weights,
    scaling,
    stat_func,
    island_weight,
    alternative,
    plan,
    keep,
    n_jobs,
    focal_canonical,
):
    """Parallel sweep over plan blocks with an order-independent reduction."""
    from joblib import Parallel, delayed, parallel_backend

    n = z.shape[0]
    greater, lesser, rlocals = _empty_tail_state(
        n, plan.permutations, keep
    )
    with parallel_backend("loky", inner_max_num_threads=1):
        worker_out = Parallel(n_jobs=n_jobs)(
            delayed(_plan_block_worker)(
                block_index,
                plan.block_bounds(block_index)[0],
                z,
                observed,
                cardinalities,
                self_weights,
                other_weights,
                scaling,
                stat_func,
                island_weight,
                alternative,
                plan,
                keep,
                focal_canonical,
            )
            for block_index in range(plan.n_blocks)
        )
    for start, b_greater, b_lesser, b_rlocals in worker_out:
        greater += b_greater
        lesser += b_lesser
        if keep:
            stop = start + b_rlocals.shape[1]
            rlocals[:, start:stop] = b_rlocals
    p_sims = _finalize_pvalues(
        greater, lesser, observed, rlocals, alternative, plan.permutations
    )
    return p_sims, rlocals if keep else np.empty((1, 1))


#######################################################################
#                   Local statistical functions                       #
#######################################################################


@njit(fastmath=False)
def _prepare_univariate(i, z, permuted_ids, weights_i):
    mask = np.ones_like(z, dtype=boolean)
    mask[i] = False
    z_no_i = z[mask]
    cardinality = len(weights_i)
    flat_permutation_ids = permuted_ids[:, :cardinality].flatten()
    zrand = z_no_i[flat_permutation_ids].reshape(-1, cardinality)
    return z[i], zrand


@njit(fastmath=False)
def _prepare_bivariate(i, z, permuted_ids, weights_i):
    zx = z[:, 0]
    zy = z[:, 1]

    cardinality = len(weights_i)

    mask = np.ones_like(zx, dtype=boolean)
    mask[i] = False
    zx_no_i = zy[mask]
    zy_no_i = zx[mask]

    flat_permutation_indices = permuted_ids[:, :cardinality].flatten()

    zxrand = zx_no_i[flat_permutation_indices].reshape(-1, cardinality)
    zyrand = zy_no_i[flat_permutation_indices].reshape(-1, cardinality)

    return zx[i], zxrand, zy[i], zyrand


@njit(fastmath=True)
def local(i, z, permuted_ids, weights_i, scaling):
    raise NotImplementedError
    # returns (k_permutations,) array of random statistics for observation i
