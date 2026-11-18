"""
Tests for the shared, stable-region-ID-bound permutation plan machinery.

These tests cover the four guarantees of the plan design:

* the same plan drives identical draws in serial and parallel execution and
  under any analysis-order reordering of the input rows;
* two statistics consuming one plan share the same random world;
* incompatible plans (candidate population, weights, transform, count, seed
  arguments) are rejected before any randomisation work begins;
* blocks are generated on demand with a bounded cache, so the full
  simulation cube is never materialised;
* the existing significance tail rules are unchanged.
"""

import pickle

import numpy as np
import pytest

from esda.crand import crand
from esda.geary_local import _local_geary
from esda.moran import Moran_Local, _moran_local_crand
from esda.permutation_plan import (
    PLAN_WEIGHTS_VERSION,
    PermutationPlan,
)
from esda.significance import calculate_significance

try:
    import joblib  # noqa: F401

    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False


@pytest.fixture
def grid_world():
    from libpysal.weights import W

    n = 36
    neighbors = {}
    for row in range(6):
        for col in range(6):
            i = row * 6 + col
            nb = []
            if row > 0:
                nb.append(i - 6)
            if row < 5:
                nb.append(i + 6)
            if col > 0:
                nb.append(i - 1)
            if col < 5:
                nb.append(i + 1)
            neighbors[i] = nb
    ids = np.array([f"reg{i}" for i in range(n)])
    neigh = {ids[i]: [ids[j] for j in neighbors[i]] for i in range(n)}
    weights = {ids[i]: [1.0] * len(neighbors[i]) for i in range(n)}
    w = W(neigh, weights, id_order=list(ids))
    rng = np.random.default_rng(0)
    z = rng.normal(size=n)
    observed = rng.normal(size=n)
    return w, ids, z, observed


def make_plan(w, permutations=80, seed=42, block_size=16):
    return PermutationPlan.from_weights(
        w, permutations, seed, block_size=block_size
    )


class TestPlanConstruction:
    def test_records_design_metadata(self, grid_world):
        w, ids, _, _ = grid_world
        plan = make_plan(w)
        assert plan.n == len(ids)
        assert plan.permutations == 80
        assert plan.seed == 42
        assert plan.block_size == 16
        assert plan.focal_exclusion is True
        assert plan.weights_version == PLAN_WEIGHTS_VERSION
        assert plan.weights_fingerprint is not None
        # canonical order is defined by the stable IDs, independent of input
        np.testing.assert_array_equal(
            plan.canonical_region_ids, np.sort(ids)
        )

    def test_requires_explicit_seed(self, grid_world):
        w, _, _, _ = grid_world
        with pytest.raises(ValueError, match="explicit `seed`"):
            PermutationPlan(
                n=w.n,
                max_cardinality=4,
                permutations=10,
                seed=None,
            )

    def test_rejects_bad_cardinality(self, grid_world):
        w, _, _, _ = grid_world
        with pytest.raises(ValueError, match="max_cardinality"):
            PermutationPlan(
                n=w.n,
                max_cardinality=w.n,  # must be <= n - 1 under focal exclusion
                permutations=10,
                seed=1,
            )

    def test_rejects_duplicate_region_ids(self, grid_world):
        w, ids, _, _ = grid_world
        dup = ids.copy()
        dup[1] = dup[0]
        with pytest.raises(ValueError, match="unique identifiers"):
            PermutationPlan(
                n=w.n,
                max_cardinality=4,
                permutations=10,
                seed=1,
                region_ids=dup,
            )

    def test_block_tiling_covers_all_permutations(self, grid_world):
        w, _, _, _ = grid_world
        plan = make_plan(w, permutations=100, block_size=32)
        assert plan.n_blocks == 4  # 32 + 32 + 32 + 4
        bounds = [plan.block_bounds(k) for k in range(plan.n_blocks)]
        assert bounds[0][0] == 0
        assert bounds[-1][1] == 100
        assert all(
            bounds[k][1] == bounds[k + 1][0]
            for k in range(plan.n_blocks - 1)
        )
        # the final block is shorter rather than padded
        last = plan.get_block(plan.n_blocks - 1)
        assert last.shape == (4, plan.max_cardinality)


class TestBlockDeterminism:
    def test_blocks_are_reproducible(self, grid_world):
        w, _, _, _ = grid_world
        first = make_plan(w).get_block(0)
        second = make_plan(w).get_block(0)
        np.testing.assert_array_equal(first, second)

    def test_blocks_independent_across_block_index(self, grid_world):
        w, _, _, _ = grid_world
        plan = make_plan(w, block_size=16)  # 80 permutations -> 5 blocks
        assert plan.n_blocks > 1
        b0 = plan.get_block(0)
        b1 = plan.get_block(1)
        assert not np.array_equal(b0, b1)

    def test_draws_respect_population_and_without_replacement(
        self, grid_world
    ):
        w, _, _, _ = grid_world
        plan = make_plan(w)
        block = plan.get_block(0)
        assert block.dtype == np.int64
        assert block.max() <= w.n - 2  # positions in the n - 1 candidate set
        for row in block:
            assert len(np.unique(row)) == len(row)

    def test_blocks_identical_after_pickle_roundtrip(self, grid_world):
        w, _, _, _ = grid_world
        plan = make_plan(w)
        restored = pickle.loads(pickle.dumps(plan))
        for k in range(plan.n_blocks):
            np.testing.assert_array_equal(
                plan.get_block(k), restored.get_block(k)
            )

    def test_lru_cache_is_bounded(self, grid_world):
        w, _, _, _ = grid_world
        plan = PermutationPlan.from_weights(
            w, 1000, seed=1, block_size=64, max_cached_blocks=2
        )
        for k in range(plan.n_blocks):
            plan.get_block(k)
        assert len(plan._block_cache) <= 2

    def test_full_cube_never_stored(self, grid_world):
        w, _, _, _ = grid_world
        plan = PermutationPlan.from_weights(
            w, 10_000, seed=1, block_size=64, max_cached_blocks=1
        )
        for start, block in plan.iter_blocks():
            assert block.shape[0] <= 64
        assert len(plan._block_cache) <= 1


class TestSerialParallelAndReorder:
    def test_serial_matches_parallel(self, grid_world):
        w, _, z, observed = grid_world
        p1, r1 = crand(
            z,
            w,
            observed,
            80,
            True,
            1,
            _moran_local_crand,
            alternative="two-sided",
            permutation_plan=make_plan(w),
        )
        p2, r2 = crand(
            z,
            w,
            observed,
            80,
            True,
            2,
            _moran_local_crand,
            alternative="two-sided",
            permutation_plan=make_plan(w),
        )
        np.testing.assert_array_equal(p1, p2)
        np.testing.assert_array_equal(r1, r2)

    def test_keep_false_matches_keep_true_pvalues(self, grid_world):
        w, _, z, observed = grid_world
        p_keep, _ = crand(
            z,
            w,
            observed,
            80,
            True,
            1,
            _moran_local_crand,
            alternative="two-sided",
            permutation_plan=make_plan(w),
        )
        p_stream, rlocals = crand(
            z,
            w,
            observed,
            80,
            False,
            1,
            _moran_local_crand,
            alternative="two-sided",
            permutation_plan=make_plan(w),
        )
        np.testing.assert_array_equal(p_keep, p_stream)
        # the streaming path returns the legacy empty placeholder, not a cube
        assert rlocals.shape == (1, 1)

    def test_invariant_under_id_reordering(self, grid_world):
        from libpysal.weights import W

        w, ids, z, observed = grid_world
        rng = np.random.default_rng(7)
        perm = rng.permutation(len(ids))
        w_reordered = W(
            w.neighbors,
            w.weights,
            id_order=list(ids[perm]),
        )
        z_reordered, obs_reordered = z[perm], observed[perm]

        p1, r1 = crand(
            z,
            w,
            observed,
            80,
            True,
            1,
            _moran_local_crand,
            alternative="two-sided",
            permutation_plan=make_plan(w),
        )
        p2, r2 = crand(
            z_reordered,
            w_reordered,
            obs_reordered,
            80,
            True,
            1,
            _moran_local_crand,
            alternative="two-sided",
            permutation_plan=make_plan(w_reordered),
        )
        # align both outputs by stable region identifier
        def to_canonical(values, weights):
            index = np.array(
                [list(weights.id_order).index(rid) for rid in ids]
            )
            return values[index]

        np.testing.assert_allclose(
            to_canonical(p2, w_reordered), to_canonical(p1, w)
        )
        np.testing.assert_allclose(
            to_canonical(r2, w_reordered), to_canonical(r1, w)
        )

    @pytest.mark.skipif(not HAS_JOBLIB, reason="joblib required")
    def test_invariant_under_id_reordering_parallel(self, grid_world):
        from libpysal.weights import W

        w, ids, z, observed = grid_world
        rng = np.random.default_rng(7)
        perm = rng.permutation(len(ids))
        w_reordered = W(
            w.neighbors, w.weights, id_order=list(ids[perm])
        )
        p1, r1 = crand(
            z,
            w,
            observed,
            80,
            True,
            1,
            _moran_local_crand,
            alternative="two-sided",
            permutation_plan=make_plan(w),
        )
        p2, r2 = crand(
            z[perm],
            w_reordered,
            observed[perm],
            80,
            True,
            3,
            _moran_local_crand,
            alternative="two-sided",
            permutation_plan=make_plan(w_reordered),
        )
        index = np.array(
            [list(w_reordered.id_order).index(rid) for rid in ids]
        )
        np.testing.assert_allclose(p2[index], p1)
        np.testing.assert_allclose(r2[index], r1)

    def test_two_statistics_share_one_plan(self, grid_world):
        w, _, z, observed = grid_world
        plan_moran = make_plan(w)
        plan_geary = make_plan(w)
        # identical plans produce identical underlying draws even though the
        # statistic functions differ
        _, r_moran = crand(
            z,
            w,
            observed,
            80,
            True,
            1,
            _moran_local_crand,
            alternative="two-sided",
            permutation_plan=plan_moran,
        )
        _, r_geary = crand(
            z,
            w,
            observed,
            80,
            True,
            1,
            _local_geary,
            alternative="two-sided",
            permutation_plan=plan_geary,
        )
        assert r_moran.shape == r_geary.shape
        # the cached blocks shared between the two plans match exactly
        for block_index in range(plan_moran.n_blocks):
            np.testing.assert_array_equal(
                plan_moran.get_block(block_index),
                plan_geary.get_block(block_index),
            )


class TestEarlyRejection:
    def test_seed_and_plan_together_rejected(self, grid_world):
        w, _, z, observed = grid_world
        with pytest.raises(ValueError, match="not both"):
            crand(
                z,
                w,
                observed,
                80,
                True,
                1,
                _moran_local_crand,
                seed=1,
                permutation_plan=make_plan(w),
            )

    def test_permutation_count_mismatch_rejected(self, grid_world):
        w, _, z, observed = grid_world
        with pytest.raises(ValueError, match="permutations but"):
            crand(
                z,
                w,
                observed,
                40,
                True,
                1,
                _moran_local_crand,
                permutation_plan=make_plan(w, permutations=80),
            )

    def test_wrong_sample_size_rejected(self, grid_world):
        w, _, z, observed = grid_world
        with pytest.raises(ValueError, match="n="):
            make_plan(w).check_compatible(w, z[:10])

    def test_changed_edge_rejected(self, grid_world):
        from libpysal.weights import W

        w, ids, z, _ = grid_world
        neighbors = {
            key: list(value) for key, value in w.neighbors.items()
        }
        weights = {key: list(value) for key, value in w.weights.items()}
        neighbors[ids[0]] = neighbors[ids[0]] + [ids[2]]
        weights[ids[0]] = weights[ids[0]] + [1.0]
        w_changed = W(neighbors, weights, id_order=list(ids))
        with pytest.raises(ValueError, match="fingerprint mismatch"):
            make_plan(w).check_compatible(w_changed, z)

    def test_changed_weight_value_rejected(self, grid_world):
        from libpysal.weights import W

        w, ids, z, _ = grid_world
        weights = {
            key: [v * 2.0 for v in value]
            for key, value in w.weights.items()
        }
        w_changed = W(w.neighbors, weights, id_order=list(ids))
        with pytest.raises(ValueError, match="fingerprint mismatch"):
            make_plan(w).check_compatible(w_changed, z)

    def test_different_id_population_rejected(self, grid_world):
        from libpysal.weights import W

        w, ids, z, _ = grid_world
        other_ids = np.array([f"other{i}" for i in range(len(ids))])
        neigh = {
            other_ids[i]: [other_ids[int(k[3:])] for k in w.neighbors[ids[i]]]
            for i in range(len(ids))
        }
        ww = {
            other_ids[i]: list(w.weights[ids[i]])
            for i in range(len(ids))
        }
        w_other = W(neigh, ww, id_order=list(other_ids))
        with pytest.raises(ValueError, match="candidate population"):
            make_plan(w).check_compatible(w_other, z)

    def test_transform_mismatch_rejected(self, grid_world):
        w, _, z, _ = grid_world
        plan = make_plan(w)
        w.transform = "r"
        with pytest.raises(ValueError, match="transform"):
            plan.check_compatible(w, z)

    def test_non_plan_object_rejected(self, grid_world):
        w, _, z, observed = grid_world
        with pytest.raises(TypeError, match="PermutationPlan instance"):
            crand(
                z,
                w,
                observed,
                80,
                True,
                1,
                _moran_local_crand,
                permutation_plan={"seed": 1},
            )


class TestTailRulesUnchanged:
    @pytest.mark.parametrize(
        "alternative",
        ["two-sided", "greater", "lesser", "directed", "folded"],
    )
    def test_plan_pvalues_match_significance_module(
        self, grid_world, alternative
    ):
        w, _, z, observed = grid_world
        p_sim, rlocals = crand(
            z,
            w,
            observed,
            80,
            True,
            1,
            _moran_local_crand,
            alternative=alternative,
            permutation_plan=make_plan(w),
        )
        reference = calculate_significance(
            observed, rlocals, alternative=alternative
        )
        np.testing.assert_allclose(p_sim, reference)
        assert p_sim.dtype == np.float32


class TestClassIntegration:
    def test_moran_local_consumes_plan_serial_and_parallel(self, grid_world):
        w, _, z, _ = grid_world
        w.transform = "r"
        plan_kwargs = dict(permutations=80, seed=42, block_size=16)
        serial = Moran_Local(
            z,
            w,
            permutations=80,
            n_jobs=1,
            alternative="two-sided",
            permutation_plan=PermutationPlan.from_weights(w, **plan_kwargs),
        )
        parallel = Moran_Local(
            z,
            w,
            permutations=80,
            n_jobs=2,
            alternative="two-sided",
            permutation_plan=PermutationPlan.from_weights(w, **plan_kwargs),
        )
        np.testing.assert_array_equal(serial.p_sim, parallel.p_sim)
        np.testing.assert_array_equal(serial.sim, parallel.sim)

    def test_legacy_seed_path_unchanged(self, grid_world):
        w, _, z, _ = grid_world
        w.transform = "r"
        first = Moran_Local(
            z, w, permutations=99, seed=12345, alternative="two-sided"
        )
        second = Moran_Local(
            z, w, permutations=99, seed=12345, alternative="two-sided"
        )
        np.testing.assert_array_equal(first.p_sim, second.p_sim)

    def test_describe_is_auditable(self, grid_world):
        w, ids, _, _ = grid_world
        plan = make_plan(w)
        summary = plan.describe()
        for key in (
            "n",
            "permutations",
            "seed",
            "block_size",
            "n_blocks",
            "focal_exclusion",
            "region_ids",
            "weights_version",
            "weights_fingerprint",
            "weights_transform",
            "max_cardinality",
        ):
            assert key in summary
        assert summary["region_ids"] == list(ids)
        # JSON-serialisable
        import json

        json.dumps(summary)


class TestGraphSupport:
    def test_graph_and_w_share_the_same_random_world(self, grid_world):
        from libpysal.graph import Graph

        w, _, z, _ = grid_world
        w.transform = "r"
        graph = Graph.from_W(w).transform("r")
        plan_w = PermutationPlan.from_weights(
            w, 80, seed=42, block_size=16
        )
        plan_graph = PermutationPlan.from_weights(
            graph, 80, seed=42, block_size=16
        )
        # identical canonical draws regardless of container type
        for block_index in range(plan_w.n_blocks):
            np.testing.assert_array_equal(
                plan_w.get_block(block_index),
                plan_graph.get_block(block_index),
            )

    def test_graph_consumer_serial_parallel(self, grid_world):
        from libpysal.graph import Graph

        from esda import Geary_Local

        w, _, z, _ = grid_world
        graph = Graph.from_W(w).transform("r")
        serial = Geary_Local(
            connectivity=graph,
            permutations=80,
            n_jobs=1,
            alternative="two-sided",
            permutation_plan=PermutationPlan.from_weights(
                graph, 80, seed=42, block_size=16
            ),
        ).fit(z)
        parallel = Geary_Local(
            connectivity=graph,
            permutations=80,
            n_jobs=2,
            alternative="two-sided",
            permutation_plan=PermutationPlan.from_weights(
                graph, 80, seed=42, block_size=16
            ),
        ).fit(z)
        np.testing.assert_allclose(serial.p_sim, parallel.p_sim)
