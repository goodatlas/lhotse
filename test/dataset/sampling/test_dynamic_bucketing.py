import random

from lhotse import CutSet
from lhotse.dataset.sampling.dynamic_bucketing import (
    DynamicBucketingSampler,
    estimate_duration_buckets,
    dynamic_bucketing,
)
from lhotse.testing.dummies import DummyManifest


def test_estimate_duration_buckets_2b():
    cuts = DummyManifest(CutSet, begin_id=0, end_id=10)
    for i, c in enumerate(cuts):
        if i < 5:
            c.duration = 1
        else:
            c.duration = 2

    bins = estimate_duration_buckets(cuts, num_buckets=2)

    assert bins == [2]


def test_estimate_duration_buckets_4b():
    cuts = DummyManifest(CutSet, begin_id=0, end_id=20)
    for i, c in enumerate(cuts):
        if i < 5:
            c.duration = 1
        elif i < 10:
            c.duration = 2
        elif i < 15:
            c.duration = 3
        elif i < 20:
            c.duration = 4

    bins = estimate_duration_buckets(cuts, num_buckets=4)

    assert bins == [2, 3, 4]


def test_dynamic_bucketing_drop_last_false():
    cuts = DummyManifest(CutSet, begin_id=0, end_id=10)
    for i, c in enumerate(cuts):
        if i < 5:
            c.duration = 1
        else:
            c.duration = 2
    rng = random.Random(0)

    sampler = dynamic_bucketing(cuts, duration_bins=[2], max_duration=5, rng=rng)
    batches = [b for b in sampler]
    sampled_cuts = [c for b in batches for c in b]

    # Invariant: no duplicated cut IDs
    assert len(set(c.id for b in batches for c in b)) == len(sampled_cuts)

    # Same number of sampled and source cuts.
    assert len(sampled_cuts) == len(cuts)

    # We sampled 4 batches with this RNG, like the following:
    assert len(batches) == 4

    assert len(batches[0]) == 2
    assert sum(c.duration for c in batches[0]) == 4

    assert len(batches[1]) == 2
    assert sum(c.duration for c in batches[1]) == 4

    assert len(batches[2]) == 5
    assert sum(c.duration for c in batches[2]) == 5

    assert len(batches[3]) == 1
    assert sum(c.duration for c in batches[3]) == 2


def test_dynamic_bucketing_drop_last_true():
    cuts = DummyManifest(CutSet, begin_id=0, end_id=10)
    for i, c in enumerate(cuts):
        if i < 5:
            c.duration = 1
        else:
            c.duration = 2
    rng = random.Random(0)

    sampler = dynamic_bucketing(
        cuts, duration_bins=[2], max_duration=5, rng=rng, drop_last=True
    )
    batches = [b for b in sampler]
    sampled_cuts = [c for b in batches for c in b]

    # Invariant: no duplicated cut IDs.
    assert len(set(c.id for b in batches for c in b)) == len(sampled_cuts)

    # Some cuts were not sampled due to drop_last.
    assert len(sampled_cuts) < len(cuts)
    assert len(sampled_cuts) == 9

    # We sampled 3 batches with this RNG, like the following:
    assert len(batches) == 3

    assert len(batches[0]) == 2
    assert sum(c.duration for c in batches[0]) == 4

    assert len(batches[1]) == 2
    assert sum(c.duration for c in batches[1]) == 4

    assert len(batches[2]) == 5
    assert sum(c.duration for c in batches[2]) == 5


def test_dynamic_bucketing_sampler():
    cuts = DummyManifest(CutSet, begin_id=0, end_id=10)
    for i, c in enumerate(cuts):
        if i < 5:
            c.duration = 1
        else:
            c.duration = 2

    sampler = DynamicBucketingSampler(cuts, max_duration=5, num_buckets=2, seed=0)
    batches = [b for b in sampler]
    sampled_cuts = [c for b in batches for c in b]

    # Invariant: no duplicated cut IDs
    assert len(set(c.id for b in batches for c in b)) == len(sampled_cuts)

    # Same number of sampled and source cuts.
    assert len(sampled_cuts) == len(cuts)

    # We sampled 4 batches with this RNG, like the following:
    assert len(batches) == 4

    assert len(batches[0]) == 2
    assert sum(c.duration for c in batches[0]) == 4

    assert len(batches[1]) == 2
    assert sum(c.duration for c in batches[1]) == 4

    assert len(batches[2]) == 5
    assert sum(c.duration for c in batches[2]) == 5

    assert len(batches[3]) == 1
    assert sum(c.duration for c in batches[3]) == 2


def test_dynamic_bucketing_sampler_filter():
    cuts = DummyManifest(CutSet, begin_id=0, end_id=10)
    for i, c in enumerate(cuts):
        if i < 5:
            c.duration = 1
        else:
            c.duration = 2

    sampler = DynamicBucketingSampler(cuts, max_duration=5, num_buckets=2, seed=0)
    sampler.filter(lambda cut: cut.duration > 1)
    batches = [b for b in sampler]
    sampled_cuts = [c for b in batches for c in b]

    # Invariant: no duplicated cut IDs
    assert len(set(c.id for b in batches for c in b)) == len(sampled_cuts)

    # Same number of sampled and source cuts.
    assert len(sampled_cuts) < len(cuts)
    assert len(sampled_cuts) == 5

    # We sampled 4 batches with this RNG, like the following:
    assert len(batches) == 3

    assert len(batches[0]) == 2
    assert sum(c.duration for c in batches[0]) == 4

    assert len(batches[1]) == 2
    assert sum(c.duration for c in batches[1]) == 4

    assert len(batches[2]) == 1
    assert sum(c.duration for c in batches[2]) == 2


def test_dynamic_bucketing_sampler_shuffle():
    cuts = DummyManifest(CutSet, begin_id=0, end_id=10)
    for i, c in enumerate(cuts):
        if i < 5:
            c.duration = 1
        else:
            c.duration = 2

    sampler = DynamicBucketingSampler(
        cuts, max_duration=5, num_buckets=2, seed=0, shuffle=True
    )

    epoch_batches = []
    for epoch in range(2):
        sampler.set_epoch(epoch)

        batches = [b for b in sampler]
        sampled_cuts = [c for b in batches for c in b]

        # Invariant: no duplicated cut IDs
        assert len(set(c.id for b in batches for c in b)) == len(sampled_cuts)

        # Invariant: Same number of sampled and source cuts.
        assert len(sampled_cuts) == len(cuts)

        epoch_batches.append(batches)

    # Epoch 0 batches are different than epoch 1 batches
    assert epoch_batches[0] != epoch_batches[1]
