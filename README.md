# torchfields
A [PyTorch](https://github.com/pytorch/pytorch) add-on for working with image mappings and displacement fields, including Spatial Transformers

Torchfields provides an abstraction that neatly encapsulates the functionality of displacement fields
as used in [Spatial Transformer Networks](https://arxiv.org/abs/1506.02025) and [Optical Flow Estimation](https://en.wikipedia.org/wiki/Optical_flow).

Fields can be treated as normal PyTorch tensors for most
purposes, and also include additional functionality for composing
displacements and sampling from tensors.

### Installation

To install torchfields simply do

```
pip install torchfields
```


### Introduction

A **displacement field** represents a *mapping* or *flow* that indicates how an image should be warped.

It is essentially a spatial tensor containing displacement vectors at each pixel, where each displacement vector indicates the displacement distance and direction at that pixel.


#### Displacement field conventions

##### Units

The standard unit of displacement is a **half-image**, so a displacement vector of magnitude 2 means that the displacement distance is equal to the side length of the displaced image. 

**Note**: *This convention originates from the original [Spatial Transformer Networks](https://arxiv.org/abs/1506.02025) paper where such fields were presented as mappings, with -1 representing the left or top edge of the image, and +1 representing the right or bottom edge.*

`torchfields` also supports seamlessly converting to and from units of **pixels** using the `pixels()` and `from_pixels()` functions.

##### Displacement direction

The most common way to warp an image by a displacement field is by sampling from it at the points pointed to by the field vectors.
This is often referred to as the **Eulerian** or **pull** convention, since the vectors in the field point to the locations from which the image should be *pulled*.
This is achieved by calling the `sample()` function (which in fact wraps PyTorch's built-in `grid_sample()`, while converting the conventions as necessary).

An alternative way to warp an image by a displacement field is by sending each pixel of the image along the corresponding displacement vector to its new location. This is referred to as the **Lagrangian** or **push** convention, since the vectors of the field indicate where an image pixel should be *pushed* to. This direction, while seemingly intuitive, is much less straight-forward to implement, since there is no definitive way to handle the discretization (for instance, what to do when the destinations are not whole pixel coordinates, when two sources map to the same destination, and when nothing maps into a destination pixel).
The solution for warping in the Lagrangian direction is to **first invert the field** using `inverse()`, and then warp the image normally using `sample()`.

*To read more about the two ways to describe flow fields, see the [Wikipedia article](https://en.wikipedia.org/wiki/Lagrangian_and_Eulerian_specification_of_the_flow_field) on the subject.*


#### Relationship to PyTorch tensors

Displacement fields inherit from `torch.Tensor`, so all functionality from [PyTorch](https://github.com/pytorch/pytorch) tensors also works with displacement fields. That is, any PyTorch function that accepts a `torch.Tensor` type will also implicitly accept a `torchfields` displacement field.

Furthermore, the module installs itself (through monkey patching) as 

```python
torch.Field
```

mirroring the `torch.Tensor` module, and all the functionality of the `torchfields` package can be conveniently accessed through that shortcut. This shortcut gets activated at the first import (using `import torchfields`).

Note, however, that the `torchfields` package is neither endorsed by nor maintained by the PyTorch developer community, and is instead a separate project maintained by researchers at Princeton University.



### Tutorial

To learn more and get started with using `torchfields` check out the [tutorial](https://colab.research.google.com/drive/1KrUjFbWjwwnsyNFTpNCZjjIJyMUP8eFx).
