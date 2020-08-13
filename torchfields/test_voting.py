import pytest
import torch
import torchfields

def test_vote_shape():
    assert(False)

def test_voting_subsets():
    assert(False)

def test_voting_weights():
    assert(False)

def test_vote():
    assert(False)

def test_vote_with_variances():
    f = torch.zeros((2,2,1,1)).field()
    f[1] = 1
    v = torch.zeros((2,1,1,1))
    vf = f.vote_with_variances(var=v, softmin_temp=1, blur_sigma=0, subset_size=1)
    tf = torch.ones((1,2,1,1)).field() / 2.
    assert(torch.equal(tf, vf))
    v[1] = 1
    # softmin_temp root of (e^(-sqrt(2)/x))/(e^(-sqrt(2)/x)+e^(0/x))-0.25=0
    vf = f.vote_with_variances(var=v, softmin_temp=1.28727, blur_sigma=0, subset_size=1)
    tf = torch.ones((1,2,1,1)).field() / 4.
    assert(torch.allclose(tf, vf))

    
