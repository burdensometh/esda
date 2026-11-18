"""
Stable, shareable permutation plans for conditional randomisation.

A :class:`PermutationPlan` binds a conditional-randomisation design to the
*stable region identifiers* of a weights object rather than to the (possibly
reordered or parallel-sharded) row positions of an array.  Several statistics
that consume the same plan then condition on exactly the same random world:

* the candidate population (the full set of region IDs),
* the focal exclusion rule (the focal region is never drawn into its own
  candidate set),
* the random seed and the block layout used to derive independent random
  streams, and
* a versioned fingerprint of the weights structure and its values, so that a
  plan built for one weights object is rejected early for an incompatible one.

Permutation blocks are generated on demand from :class:`numpy.random.SeedSequence`
spawning, so the full ``(permutations, max_cardinality)`` simulation cube is
never materialised or cached as a whole: only a small, bounded number of
blocks is held in memory at any time.  Block content is a deterministic
function of ``(seed, block_index, n_minus_one)`` and is therefore identical in
serial and parallel execution and independent of how sites are chunked.
"""

from collections import OrderedDict
from hashlib import blake2b

import numpy as np

__all__ = ["PermutationPlan", "PLAN_WEIGHTS_VERSION"]

#: Version of the weights fingerprint embedded in every plan.  Bump this when
#: the hashing convention changes; plans carrying an older version are
#: rejected rather than silently reinterpreted.
PLAN_WEIGHTS_VERSION = 1

#: Default number of permutation draws generated per on-demand block.
DEFAULT_BLOCK_SIZE = 128

#: Default maximum number of blocks retained in the per-plan LRU cache.
DEFAULT_MAX_CACHED_BLOCKS = 4


def _fingerprint_weights(adjacency, region_ids, weights_version=PLAN_WEIGHTS_VERSION):
    """
    Hash the structure *and* the values of a CSR adjacency matrix.

    Triples are expressed in stable region identifiers rather than row /
    column positions, so the hash is invariant to any reordering of the
    analysis order while still detecting any change in connectivity or in a
    weight value.
    """
    region_ids = np.asarray(region_ids)
    coo = adjacency.tocoo()
    # Translate positional triples into stable-ID triples.
    rows = region_ids[coo.row]
    cols = region_ids[coo.col]
    order = np.lexsort((cols, rows))
    hasher = blake2b(digest_size=16)
    hasher.update(b"esda-permutation-plan-v%d" % weights_version)
    # Hash the string form of the identifiers so any hashable ID dtype
    # (int, str, tuple-like objects) is supported without truncation.
    hasher.update(
        np.asarray(rows[order], dtype=object)
        .astype(str)
        .tobytes()
    )
    hasher.update(
        np.asarray(cols[order], dtype=object)
        .astype(str)
        .tobytes()
    )
    hasher.update(
        np.asarray(coo.data[order], dtype=np.float64).tobytes()
    )
    return hasher.hexdigest()


class PermutationPlan:
    """
    A stable conditional-randomisation design shared across statistics.

    The plan records everything needed to reproduce a permutation draw without
    holding the draws themselves:

    * ``region_ids`` -- the candidate population, i.e. the stable IDs of every
      region in analysis order.  Draws index the *positions* of this array;
      the focal region is excluded at consumption time, so the same draw
      applies unchanged when the input rows are reordered.
    * ``n``, ``max_cardinality`` and ``permutations`` -- the design shape.
    * ``seed`` -- the root seed.  Every block gets an independent child
      stream derived from it via :class:`numpy.random.SeedSequence`.
    * ``block_size`` -- the on-demand generation granularity; the full
      permutation cube is never stored.
    * ``weights_version`` and ``weights_fingerprint`` -- a versioned hash of
      the weights structure/values used to build the plan.

    Parameters
    ----------
    n : int
        Number of candidate regions.
    max_cardinality : int
        Largest neighbor cardinality across regions; each draw contains this
        many candidate positions (shorter neighbor sets use a prefix).
    permutations : int
        Total number of permutation draws covered by the plan.
    seed : int
        Root seed for reproducibility.
    region_ids : array-like, optional
        Stable identifiers for the ``n`` regions.  Defaults to
        ``numpy.arange(n)``.
    weights_fingerprint : str, optional
        Versioned fingerprint of the weights object.  Plans are typically
        constructed through :meth:`from_weights`; a plan without a fingerprint
        can only be consumed by callers that skip weights validation.
    weights_version : int, optional
        Fingerprint convention version.  Defaults to
        :data:`PLAN_WEIGHTS_VERSION`.
    focal_exclusion : bool, optional
        If True (default), the focal region is excluded from its own
        candidate set, matching the conditional-randomisation null.
    block_size : int, optional
        Number of draws generated per block.  Defaults to
        :data:`DEFAULT_BLOCK_SIZE`.
    max_cached_blocks : int, optional
        Maximum number of generated blocks retained in the LRU cache.
        Defaults to :data:`DEFAULT_MAX_CACHED_BLOCKS`.

    Examples
    --------
    Build a plan from a weights object and share it across two local
    statistics so both see identical random worlds:

    >>> import libpysal as lps
    >>> from esda.permutation_plan import PermutationPlan
    >>> from esda import Moran_Local, Geary_Local
    >>> w = lps.weights.lat2W(10, 10)
    >>> plan = PermutationPlan.from_weights(w, permutations=999, seed=12345)
    >>> lm = Moran_Local(y, w, permutation_plan=plan)  # doctest: +SKIP
    >>> lg = Geary_Local(connectivity=w,
    ...                  permutation_plan=plan).fit(y)  # doctest: +SKIP
    """

    def __init__(
        self,
        n,
        max_cardinality,
        permutations,
        seed,
        region_ids=None,
        weights_fingerprint=None,
        weights_version=PLAN_WEIGHTS_VERSION,
        focal_exclusion=True,
        block_size=DEFAULT_BLOCK_SIZE,
        max_cached_blocks=DEFAULT_MAX_CACHED_BLOCKS,
    ):
        n = int(n)
        max_cardinality = int(max_cardinality)
        permutations = int(permutations)
        block_size = int(block_size)
        if n < 2:
            raise ValueError("`n` must be at least 2 for randomisation.")
        if max_cardinality < 1:
            raise ValueError("`max_cardinality` must be at least 1.")
        if max_cardinality > n - 1:
            raise ValueError(
                "`max_cardinality` cannot exceed n - 1 because the focal "
                f"region is excluded; got {max_cardinality} for n={n}."
            )
        if permutations < 1:
            raise ValueError("`permutations` must be at least 1.")
        if block_size < 1:
            raise ValueError("`block_size` must be at least 1.")
        if max_cached_blocks < 1:
            raise ValueError("`max_cached_blocks` must be at least 1.")
        if seed is None:
            raise ValueError(
                "A shared plan requires an explicit `seed`; got None."
            )

        region_ids = np.arange(n) if region_ids is None else np.asarray(region_ids)
        if region_ids.shape != (n,):
            raise ValueError(
                f"`region_ids` must have shape ({n},); got {region_ids.shape}."
            )
        if len(np.unique(region_ids)) != n:
            raise ValueError("`region_ids` must contain unique identifiers.")

        # The canonical ordering is defined by the stable identifiers
        # themselves, not by the row order in which a weights object happened
        # to be supplied. Two plans over the same ID universe therefore
        # generate identical draws even when their inputs were reordered.
        canonical_order = np.argsort(region_ids)
        canonical_region_ids = region_ids[canonical_order]

        self.n = n
        self.max_cardinality = max_cardinality
        self.permutations = permutations
        self.seed = int(seed)
        self.region_ids = region_ids
        self.canonical_region_ids = canonical_region_ids
        self.weights_fingerprint = weights_fingerprint
        self.weights_version = int(weights_version)
        # Transformed state the plan expects at consumption time.
        self.weights_transform = None
        self.focal_exclusion = bool(focal_exclusion)
        self.block_size = block_size
        self._max_cached_blocks = int(max_cached_blocks)
        self._block_cache = OrderedDict()

    # ------------------------------------------------------------------
    # Construction / validation
    # ------------------------------------------------------------------

    @classmethod
    def from_weights(
        cls,
        w,
        permutations,
        seed,
        block_size=DEFAULT_BLOCK_SIZE,
        max_cached_blocks=DEFAULT_MAX_CACHED_BLOCKS,
        focal_exclusion=True,
        transform=None,
    ):
        """
        Build a plan bound to the stable region IDs of a weights object.

        Parameters
        ----------
        w : libpysal.weights.W
            Weights object whose ``id_order`` supplies the stable region
            identifiers and whose sparse adjacency supplies the neighbor
            structure and weight values.
        permutations : int
            Total number of permutation draws.
        seed : int
            Root seed for reproducibility.
        block_size : int, optional
            Draws generated per on-demand block.
        max_cached_blocks : int, optional
            Maximum number of blocks retained in the LRU cache.
        focal_exclusion : bool, optional
            Whether the focal region is excluded from its own candidate set.
        transform : str, optional
            Weights transformation (e.g. ``'r'`` for row-standardised) that
            consuming statistics will apply to ``w`` before randomisation.
            The plan fingerprints the transformed weights and records the
            transformation as part of its versioned design, so a plan built
            for one transform is rejected for weights carrying another.
            Defaults to the weights' current transform.

        Returns
        -------
        PermutationPlan
        """
        try:
            adjacency = w.sparse
            region_ids = np.asarray(
                w.id_order if hasattr(w, "id_order") else w.unique_ids
            )
            # W exposes the active transform as the `.transform` attribute,
            # whereas Graph exposes it through `.transformation` (`.transform`
            # is a method there).
            current_transform = getattr(
                w,
                "transformation",
                getattr(w, "transform", "b"),
            )
        except AttributeError as error:
            raise TypeError(
                "`from_weights` expects a libpysal weights object with a "
                "`sparse` adjacency and stable identifiers (`id_order` for "
                "W or `unique_ids` for Graph); construct a plan directly "
                "with PermutationPlan(...) for other inputs."
            ) from error

        if transform is None:
            transform = current_transform
        if transform != current_transform:
            # Work on a transformed copy so building a plan never mutates
            # the caller's weights object. W sets the transform in place via
            # an attribute, Graph returns a new transformed instance.
            if hasattr(w, "id_order"):
                w.transform = transform
            else:
                w = w.transform(transform)
            adjacency = w.sparse

        with np.errstate(invalid="ignore"):
            no_self = adjacency.copy()
            no_self.setdiag(0)
            no_self.eliminate_zeros()
        # CSR indptr differences give per-row nonzeros for both csr_matrix
        # (W) and csr_array (Graph), which do not share getnnz.
        max_cardinality = int(np.diff(no_self.indptr).max())
        fingerprint = _fingerprint_weights(no_self, region_ids)
        plan = cls(
            n=adjacency.shape[0],
            max_cardinality=max(1, max_cardinality),
            permutations=permutations,
            seed=seed,
            region_ids=region_ids,
            weights_fingerprint=fingerprint,
            focal_exclusion=focal_exclusion,
            block_size=block_size,
            max_cached_blocks=max_cached_blocks,
        )
        plan.weights_transform = transform
        return plan

    @property
    def n_blocks(self):
        """Number of blocks that tile the full ``permutations`` range."""
        return (self.permutations + self.block_size - 1) // self.block_size

    def block_bounds(self, block_index):
        """
        Return the half-open ``(start, stop)`` draw range of a block.

        The final block may be shorter; every other block has exactly
        ``block_size`` draws.
        """
        if not 0 <= block_index < self.n_blocks:
            raise IndexError(
                f"block index {block_index} out of range for "
                f"{self.n_blocks} blocks."
            )
        start = block_index * self.block_size
        stop = min(start + self.block_size, self.permutations)
        return start, stop

    def canonical_positions(self, id_order):
        """
        Return the local rows to fetch so that fetched rows follow the
        plan's canonical (sorted stable-ID) order.

        Parameters
        ----------
        id_order : array-like
            Stable region identifiers in the caller's array order, as given
            by ``W.id_order``.

        Returns
        -------
        numpy.ndarray
            Integer array ``rows`` such that ``data[rows]`` is aligned to
            :attr:`canonical_region_ids`; i.e. ``rows[k]`` is the local
            position of the region whose canonical position is ``k``.
        """
        id_to_canonical = {
            region_id: pos
            for pos, region_id in enumerate(self.canonical_region_ids)
        }
        position_of_each_local = np.array(
            [id_to_canonical[region_id] for region_id in id_order],
            dtype=np.int64,
        )
        return np.argsort(position_of_each_local)

    def check_compatible(self, w, z):
        """
        Validate that ``w`` and ``z`` match the design recorded in the plan.

        Raises
        ------
        ValueError
            If the candidate population, region IDs, weights fingerprint or
            array lengths do not match.  Raising *before* any randomisation
            work begins prevents two statistics from silently consuming
            different random worlds.
        """
        n = len(z)
        if n != self.n:
            raise ValueError(
                f"PermutationPlan was built for n={self.n} observations but "
                f"received data with n={n}."
            )
        if not hasattr(w, "sparse") or not hasattr(
            w, "id_order"
        ) and not hasattr(w, "unique_ids"):
            raise TypeError(
                "A weights object with a sparse adjacency and stable "
                "identifiers (`id_order` for W or `unique_ids` for Graph) "
                "is required to validate a shared PermutationPlan."
            )
        incoming_ids = np.asarray(
            w.id_order if hasattr(w, "id_order") else w.unique_ids
        )
        if incoming_ids.shape != self.region_ids.shape or not np.array_equal(
            np.sort(incoming_ids), np.sort(self.region_ids)
        ):
            raise ValueError(
                "PermutationPlan region IDs do not match the weights object's "
                "id_order: the candidate population differs."
            )
        if self.weights_transform is not None:
            active_transform = getattr(
                w,
                "transformation",
                getattr(w, "transform", None),
            )
            if active_transform != self.weights_transform:
                raise ValueError(
                    f"PermutationPlan was built with weights transform "
                    f"{self.weights_transform!r} but the supplied weights "
                    f"carry transform {active_transform!r}. Apply the same "
                    "transformation before consuming the plan or rebuild it."
                )
        if self.weights_fingerprint is not None:
            with np.errstate(invalid="ignore"):
                no_self = w.sparse.copy()
                no_self.setdiag(0)
                no_self.eliminate_zeros()
            incoming_fp = _fingerprint_weights(
                no_self,
                incoming_ids,
                weights_version=self.weights_version,
            )
            if incoming_fp != self.weights_fingerprint:
                raise ValueError(
                    "PermutationPlan weights fingerprint mismatch: the "
                    "weights structure or values differ from those the plan "
                    "was built with (weights_version="
                    f"{self.weights_version}, transform="
                    f"{self.weights_transform!r}). Rebuild the plan for "
                    "these weights rather than reusing the incompatible one."
                )

    # ------------------------------------------------------------------
    # On-demand block generation
    # ------------------------------------------------------------------

    def _generate_block(self, block_index):
        start, stop = self.block_bounds(block_index)
        draws = stop - start
        # Mixing the block index into the seed makes every block's stream
        # independent and context-free: the same block is produced in serial
        # sweeps, parallel workers and unpickled copies of the plan, and the
        # result does not depend on how many other blocks were generated
        # before it.
        child = np.random.SeedSequence([self.seed, block_index])
        rng = np.random.default_rng(child)
        # Draws index the n - 1 positions that remain after focal exclusion;
        # the focal row itself is removed by the consumer. One draw per row
        # mirrors the conditional-randomisation semantics of vec_permutations
        # and avoids numpy's prod(size) > population restriction on a single
        # multidimensional choice.
        block = np.empty((draws, self.max_cardinality), dtype=np.int64)
        for row in range(draws):
            block[row] = rng.choice(
                self.n - 1, size=self.max_cardinality, replace=False
            )
        return block

    def get_block(self, block_index):
        """
        Return one permutation block as a ``(draws, max_cardinality)`` array.

        Blocks are generated on demand and retained in a bounded LRU cache,
        so repeatedly consuming the same plan (e.g. from several statistics)
        reuses draws without ever storing the full simulation cube.
        """
        block = self._block_cache.get(block_index)
        if block is not None:
            self._block_cache.move_to_end(block_index)
            return block
        block = self._generate_block(block_index)
        self._block_cache[block_index] = block
        while len(self._block_cache) > self._max_cached_blocks:
            self._block_cache.popitem(last=False)
        return block

    def iter_blocks(self):
        """
        Yield ``(start, block)`` pairs in draw order without caching the
        whole cube.  Each block is evicted under the same LRU policy as
        :meth:`get_block`, so a single sweep holds only a bounded number.
        """
        for block_index in range(self.n_blocks):
            start, _ = self.block_bounds(block_index)
            yield start, self.get_block(block_index)

    # ------------------------------------------------------------------
    # Bookkeeping
    # ------------------------------------------------------------------

    def describe(self):
        """
        Return a JSON-serialisable summary of the recorded design.

        The summary contains the candidate population, focal exclusion rule,
        seed, block layout and versioned weights information needed to audit
        which random world a statistic consumed.
        """
        return {
            "n": self.n,
            "max_cardinality": self.max_cardinality,
            "permutations": self.permutations,
            "seed": self.seed,
            "block_size": self.block_size,
            "n_blocks": self.n_blocks,
            "focal_exclusion": self.focal_exclusion,
            "region_ids": self.region_ids.tolist(),
            "weights_version": self.weights_version,
            "weights_transform": self.weights_transform,
            "weights_fingerprint": self.weights_fingerprint,
        }

    def __getstate__(self):
        # The LRU cache is a runtime convenience, not part of the design;
        # drop it on pickling so parallel workers receive a compact plan.
        state = self.__dict__.copy()
        state["_block_cache"] = OrderedDict()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __repr__(self):
        return (
            f"PermutationPlan(n={self.n}, permutations={self.permutations}, "
            f"max_cardinality={self.max_cardinality}, seed={self.seed}, "
            f"block_size={self.block_size}, n_blocks={self.n_blocks}, "
            f"weights_version={self.weights_version})"
        )
