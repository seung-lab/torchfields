import pytest
import torch
import torchfields

# def test_vote_shape():
#     assert False
#
#
# def test_voting_subsets():
#     assert False
#
#
# def test_voting_weights():
#     assert False
#
#
# def test_vote():
#     assert False


def test_vote_with_variances():
    f = torch.zeros((2, 2, 1, 1)).field()
    f[1] = 1
    v = torch.zeros((2, 1, 1, 1))
    vf = f.vote_with_variances(var=v, softmin_temp=1, blur_sigma=0, subset_size=1)
    tf = torch.ones((1, 2, 1, 1)).field() / 2.0
    assert torch.equal(tf, vf)
    v[1] = 1
    # softmin_temp root of (e^(-sqrt(2)/x))/(e^(-sqrt(2)/x)+e^(0/x))-0.25=0
    vf = f.vote_with_variances(var=v, softmin_temp=1.28727, blur_sigma=0, subset_size=1)
    tf = torch.ones((1, 2, 1, 1)).field() / 4.0
    assert torch.allclose(tf, vf)


def test_vote_with_distances():
    f = torch.zeros((2, 2, 1, 1)).field()
    f[1] = 1
    d = torch.ones((2, 1, 1))
    d[1] = 2
    df = f.vote_with_distances(distances=d, softmin_temp=1, blur_sigma=0, subset_size=1)
    tf = torch.ones((1, 2, 1, 1)).field() / 2.0
    assert torch.equal(tf, df)
    f = torch.zeros((2, 2, 1, 1)).field()
    f[1] = 1
    d = torch.ones((2, 1, 1))
    d[1] = 2
    df = f.get_vote_weights_with_distances(
        distances=d, softmin_temp=1, blur_sigma=0, subset_size=2
    )
    tf = torch.ones((2, 1, 1)) / 3.0
    tf[0] *= 2.0
    assert torch.equal(tf, df)
    f = torch.zeros((3, 2, 1, 1)).field()
    f[1] = 1
    f[2] = 1.2
    d = torch.ones((3, 1, 1))
    d[1] = 2
    d[2] = 3
    df = f.vote_with_distances(
        distances=d, softmin_temp=0.01, blur_sigma=0, subset_size=2
    )
    tf = (
        torch.ones((1, 2, 1, 1)).field() * 3 / 5.0
        + torch.ones((1, 2, 1, 1)).field() * 1.2 * 2 / 5.0
    )
    assert torch.allclose(tf, df)


def test_priority_vote():
    # if subsets are only 1, then return weights that identify highest priority
    f = torch.zeros((2, 2, 1, 1)).field()
    p = torch.ones((2, 1, 1))
    p[1] = 0
    vfw = f.get_priority_vote_weights(priorities=p, subset_size=1)
    tfw = torch.ones((2, 1, 1))
    tfw[1] = 0
    assert torch.equal(tfw, vfw)

    # v1 in consensus with v3, so return weights for v1
    f = torch.zeros((3, 2, 1, 1)).field()
    f[1] = 1
    p = torch.full((3, 1, 1), fill_value=3)
    p[1] = 2
    p[2] = 1
    vfw = f.get_priority_vote_weights(
        priorities=p, consensus_threshold=1, subset_size=2
    )
    tfw = torch.ones((3, 1, 1))
    tfw[1] = 0
    tfw[2] = 0
    assert torch.equal(tfw, vfw)

    # regardless of whether the consensus_threshold marks v2 as in consensus
    vfw = f.get_priority_vote_weights(
        priorities=p, consensus_threshold=2, subset_size=2
    )
    tfw = torch.ones((3, 1, 1))
    tfw[1] = 0
    tfw[2] = 0
    assert torch.equal(tfw, vfw)

    # v1 in consensus with v3, then v2 in consensus in v3
    f = torch.zeros((3, 2, 2, 1)).field()
    f[0, :, 1, :] = 1
    f[1, :, 0, :] = 1
    p = torch.full((3, 2, 1), fill_value=3)
    p[1] = 2
    p[2] = 1
    vfw = f.get_priority_vote_weights(
        priorities=p, consensus_threshold=1, subset_size=2
    )
    tfw = torch.ones((3, 2, 1))
    tfw[0, 1, :] = 0
    tfw[1, 0, :] = 0
    tfw[2] = 0
    assert torch.equal(tfw, vfw)

    # increasing the consensus_threshold makes v1 always in consensus
    vfw = f.get_priority_vote_weights(
        priorities=p, consensus_threshold=2, subset_size=2
    )
    tfw = torch.ones((3, 2, 1))
    tfw[1] = 0
    tfw[2] = 0
    assert torch.equal(tfw, vfw)

    # increasing the consensus_threshold makes v1 always in consensus
    vf = f.priority_vote(priorities=p, consensus_threshold=2, subset_size=2)
    tf = torch.zeros((1, 2, 2, 1))
    tf[0, :, 1, :] = 1
    assert torch.equal(tf, vf)

    # just test that blurring doesn't throw an error
    f = torch.ones((3, 2, 4, 4)).field()
    p = torch.ones((3, 4, 4))
    vf = f.priority_vote(priorities=p, consensus_threshold=2, subset_size=2)
    assert torch.allclose(f, vf)

    # consensus_threshold=0 should return v2
    f = torch.zeros((3, 2, 1, 1)).field()
    f[0] = 1
    p = torch.full((3, 1, 1), fill_value=3)
    p[1] = 2
    p[2] = 1
    vfw = f.get_priority_vote_weights(
        priorities=p, consensus_threshold=0, subset_size=2
    )
    tfw = torch.zeros((3, 1, 1))
    tfw[1] = 1
    assert torch.equal(tfw, vfw)

    # no negative consensus_threshold
    with pytest.raises(ValueError):
        vfw = f.get_priority_vote_weights(
            priorities=p, consensus_threshold=-1, subset_size=2
        )


def test_gaussian_blur():
    f = torch.ones((3, 1, 4, 4))
    gf = torchfields.voting.gaussian_blur(data=f)
    assert torch.allclose(gf, f)
