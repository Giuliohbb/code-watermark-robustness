import itertools
import math

import pytest
import torch

from cwm.watermarks.greenlist import GreenList, z_score


def test_mask_is_deterministic_and_sized_by_gamma():
    greenlist = GreenList(vocab_size=1000, gamma=0.25, hash_key=15485863)
    first = greenlist.mask(42)
    again = GreenList(vocab_size=1000, gamma=0.25, hash_key=15485863).mask(42)
    assert torch.equal(first, again)
    assert int(first.sum()) == 250


def test_mask_depends_on_previous_token_and_key():
    a = GreenList(1000, 0.5, 15485863)
    b = GreenList(1000, 0.5, 7)
    assert not torch.equal(a.mask(1), a.mask(2))
    assert not torch.equal(a.mask(1), b.mask(1))


def test_matches_markllm_seeding_scheme():
    """seed = hash_key * prev_token, green = first gamma*V entries of randperm."""
    vocab, key, prev = 500, 15485863, 17
    generator = torch.Generator(device="cpu").manual_seed(key * prev)
    expected = torch.randperm(vocab, generator=generator)[: vocab // 2]
    mask = GreenList(vocab, 0.5, key).mask(prev)
    assert set(torch.nonzero(mask).flatten().tolist()) == set(expected.tolist())


def test_green_fraction_of_random_text_is_close_to_gamma():
    greenlist = GreenList(vocab_size=2000, gamma=0.5, hash_key=3)
    generator = torch.Generator().manual_seed(0)
    tokens = torch.randint(0, 2000, (4001,), generator=generator).tolist()
    greens = sum(greenlist.is_green(p, t) for p, t in itertools.pairwise(tokens))
    assert abs(greens / 4000 - 0.5) < 0.03


def test_z_score_values():
    assert z_score(50, 100, 0.5) == 0.0
    assert z_score(70, 100, 0.5) == pytest.approx(4.0)
    assert math.isnan(z_score(0, 0, 0.5))


@pytest.mark.parametrize("gamma", [0.0, 1.0, -0.1])
def test_invalid_gamma(gamma):
    with pytest.raises(ValueError):
        GreenList(100, gamma, 1)
