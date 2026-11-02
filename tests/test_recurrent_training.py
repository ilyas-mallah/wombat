import random

from wombat.training import _sample_log_uniform_depth, _sample_progressive_segment


def test_log_uniform_depth_sampler_reaches_non_anchor_integers() -> None:
    randomizer = random.Random(4)
    samples = {_sample_log_uniform_depth(randomizer, 13) for _ in range(1000)}
    assert samples == set(range(1, 14))


def test_progressive_segment_covers_prefixes_without_exceeding_budget() -> None:
    randomizer = random.Random(7)
    samples = [_sample_progressive_segment(randomizer, 8) for _ in range(1000)]
    assert {prefix for prefix, _ in samples} == set(range(8))
    assert all(tail >= 1 and prefix + tail <= 8 for prefix, tail in samples)


def test_progressive_segment_caps_gradient_tail() -> None:
    randomizer = random.Random(3)
    samples = [_sample_progressive_segment(randomizer, 32, 4) for _ in range(1000)]
    assert all(1 <= tail <= 4 for _, tail in samples)
    assert all(prefix + tail <= 32 for prefix, tail in samples)
    assert max(prefix for prefix, _ in samples) == 31
