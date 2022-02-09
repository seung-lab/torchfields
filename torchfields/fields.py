"""PyTorch tensor type for working with displacement vector fields
"""
from typing import Any
import torch
import torch.nn.functional as F
from functools import wraps

from .utils import permute_input, permute_output, ensure_dimensions
from . import inversion
from . import voting


####################################
# DisplacementField Class Definition
####################################


class DisplacementField(torch.Tensor):
    """An abstraction that encapsulates functionality of displacement fields
    as used in Spatial Transformer Networks.

    DisplacementFields can be treated as normal PyTorch tensors for most
    purposes, and also include additional functionality for composing
    displacements and sampling from tensors.
    """

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if not all(issubclass(cls, t) for t in types):
            return NotImplemented

        return super().__torch_function__(func, types, args, kwargs)

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, *args, **kwargs):
        if len(self.shape) < 3:
            raise ValueError(
                "The displacement field must have a components "
                "dimension. Only {} dimensions are present.".format(len(self.shape))
            )
        if self.shape[-3] != 2:
            raise ValueError(
                "The displacement field must have exactly 2 "
                "components, not {}.".format(self.shape[-3])
            )

    def __repr__(self, *args, **kwargs):
        out = super().__repr__(*args, **kwargs)
        return out.replace("tensor", "field", 1).replace("\n ", "\n")

    _cache_identities = False
    _identities = {}

    @classmethod
    def _set_identity_mapping_cache(cls, mode: bool, clear_cache: bool = False) -> None:
        cls._cache_identities = mode
        if clear_cache:
            cls._identities = {}

    @classmethod
    def is_identity_mapping_cache_enabled(cls) -> bool:
        """``True`` if identity_mapping() calls are currently cached, ``else False``."""
        return cls._cache_identities

    # Conversion to and from torch.Tensor

    def field_(self, *args, **kwargs):
        """Converts a `torch.Tensor` to a `DisplacementField`

        Note: This does not make a copy, but rather modifies it in place.
            Because of this, nothing is added to the computation graph.
            To produce a new `DisplacementField` from a tensor and/or add a
            step to the computation graph, instead use `field()`,
            the not-in-place version.
        """
        allowed_types = DisplacementField.__bases__
        if not isinstance(self, allowed_types):
            raise TypeError(
                "'{}' cannot be converted to '{}'. Valid options are: {}".format(
                    type(self).__name__,
                    DisplacementField.__name__,
                    [base.__module__ + "." + base.__name__ for base in allowed_types],
                )
            )
        if len(self.shape) < 3:
            raise ValueError(
                "The displacement field must have a components "
                "dimension. Only {} dimensions are present.".format(len(self.shape))
            )
        if self.shape[-3] != 2:
            raise ValueError(
                "The displacement field must have exactly 2 "
                "components, not {}.".format(self.shape[-3])
            )
        self.__class__ = DisplacementField
        self.__init__(*args, **kwargs)  # in case future __init__ is nonempty
        return self

    torch.Tensor.field_ = field_  # adds conversion to torch.Tensor superclass
    _from_superclass = field_  # for use in `return_subclass_type()`

    def field(data, *args, **kwargs):
        """Converts a `torch.Tensor` to a `DisplacementField`
        """
        if isinstance(data, torch.Tensor):
            return DisplacementField.field_(data.clone(), *args, **kwargs)
        else:
            return DisplacementField.field_(torch.tensor(data, *args, **kwargs).float())

    torch.Tensor.field = field  # adds conversion to torch.Tensor superclass
    torch.field = field

    def tensor_(self):
        """Converts the `DisplacementField` to a standard `torch.Tensor`
        in-place

        Note: This does not make a copy, but rather modifies it in place.
            Because of this, nothing is added to the computation graph.
            To produce a new `torch.Tensor` from a `DisplacementField` and/or
            add a copy step to the computation graph, instead use `tensor()`,
            the not-in-place version.
        """
        self.__class__ = torch.Tensor
        return self

    def tensor(self):
        """Converts the `DisplacementField` to a standard `torch.Tensor`
        """
        return self.clone().tensor_()

    # Constuctors for typical displacent fields

    def identity(*args, **kwargs):
        """Returns an identity displacement field (containing all zero vectors)

        See :func:`torch.zeros`
        """
        if len(args) > 0 and isinstance(args[0], torch.Tensor):
            tensor_like, *args = args
            if "device" not in kwargs or kwargs["device"] is None:
                kwargs["device"] = tensor_like.device
            if "size" not in kwargs or kwargs["size"] is None:
                kwargs["size"] = tensor_like.shape
            if "dtype" not in kwargs or kwargs["dtype"] is None:
                kwargs["dtype"] = tensor_like.dtype
        return torch.zeros(*args, **kwargs).field_()

    zeros_like = zeros = identity

    def ones(*args, **kwargs):
        """Returns a displacement field type tensor of all ones.

        The result is a translation field of half the image in all coordinates,
        which is not usually a useful field on its own, but can be multiplied
        by a factor to get different translations.

        See :func:`torch.ones`
        """
        if len(args) > 0 and isinstance(args[0], torch.Tensor):
            tensor_like, *args = args
            if "device" not in kwargs or kwargs["device"] is None:
                kwargs["device"] = tensor_like.device
            if "size" not in kwargs or kwargs["size"] is None:
                kwargs["size"] = tensor_like.shape
            if "dtype" not in kwargs or kwargs["dtype"] is None:
                kwargs["dtype"] = tensor_like.dtype
        return torch.ones(*args, **kwargs).field_()

    ones_like = ones

    def rand(*args, **kwargs):
        """Returns a displacement field type tensor with each vector
        component randomly sampled from the uniform distribution on [0, 1).

        See :func:`torch.rand`
        """
        if len(args) > 0 and isinstance(args[0], torch.Tensor):
            tensor_like, *args = args
            if "device" not in kwargs or kwargs["device"] is None:
                kwargs["device"] = tensor_like.device
            if "size" not in kwargs or kwargs["size"] is None:
                kwargs["size"] = tensor_like.shape
            if "dtype" not in kwargs or kwargs["dtype"] is None:
                kwargs["dtype"] = tensor_like.dtype
        return torch.rand(*args, **kwargs).field_()

    rand_like = rand

    @torch.no_grad()
    def rand_in_bounds(*args, **kwargs):
        """Returns a displacement field where each displacement
        vector samples from a uniformly random location from within the
        bounds of the sampled tensor (when called with `sample()` or
        `compose()`).

        See :func:`torch.rand` for the function signature.
        """
        rand_tensor = DisplacementField.rand(*args, **kwargs)
        if not isinstance(rand_tensor, DisplacementField):
            # if incompatible, fail with the proper error
            rand_tensor = DisplacementField._from_superclass(rand_tensor)
        field = rand_tensor * 2 - 1  # rescale to [-1, 1)
        field = field - field.identity_mapping()
        return field.requires_grad_(rand_tensor.requires_grad)

    rand_in_bounds_like = rand_in_bounds

    def _get_parameters(tensor, shape=None, device=None, dtype=None, override=False):
        """Auxiliary function to deduce the right set of parameters to a tensor
        function.
        In particular, if `tensor` is a `torch.Tensor`, it uses those values.
        Otherwise, if the values are not explicitly specified, returns the
        default values.
        If `override` is set to `True`, then the parameters passed override
        those of the tensor unless they are None.
        """
        if isinstance(tensor, torch.Tensor):
            shape = shape if override and (shape is not None) else tensor.shape
            device = device if override and (device is not None) else tensor.device
            dtype = dtype if override and (dtype is not None) else tensor.dtype
        else:
            if device is None:
                try:
                    device = torch.cuda.current_device()
                except AssertionError:
                    device = "cpu"
            if dtype is None:
                dtype = torch.float
        if isinstance(shape, tuple):
            batch_dim = shape[0] if len(shape) > 3 else 1
            if len(shape) < 2:
                raise ValueError(
                    "The shape must have at least two spatial "
                    "dimensions. Recieved shape {}.".format(shape)
                )
            while len(shape) < 4:
                shape = (1,) + shape
        else:
            try:
                shape = torch.Size((1, 2, shape, shape))
                batch_dim = 1
            except TypeError:
                raise TypeError(
                    "'shape' must be an 'int', 'tuple', or "
                    "'torch.Size'. Received '{}'".format(type(shape).__qualname__)
                )
        device = torch.device(device)
        if dtype == torch.double:
            tensor_type = (
                torch.DoubleTensor if device.type == "cpu" else torch.cuda.DoubleTensor
            )
        elif dtype == torch.float:
            tensor_type = (
                torch.FloatTensor if device.type == "cpu" else torch.cuda.FloatTensor
            )
        else:
            raise ValueError(
                "The data type must be either torch.float or "
                "torch.double. Recieved {}.".format(dtype)
            )
        return {
            "shape": shape,
            "batch_dim": batch_dim,
            "device": device,
            "dtype": dtype,
            "tensor_type": tensor_type,
        }

    @torch.no_grad()
    def identity_mapping(size, device=None, dtype=None):
        """Returns an identity mapping with -1 and +1 at the corners of the
        image (not the centers of the border pixels as in PyTorch 1.1).

        Note that this is NOT an identity displacement field, and therefore
        sampling with it will not return the input.
        To get the identity displacement field, use `identity()`.
        Instead, this creates a mapping that maps each coordinate to its
        own coordinate vector (in the [-1, +1] space).

        Args:
            size: either an `int` or a `torch.Size` of the form `(N, C, H, W)`.
                `C` is ignored.
            device (torch.device): the device (cpu/cuda) on which to create
                the mapping
            dtype (torch.dtype): the data type of resulting mapping. Can be
                `torch.float` or `torch.double`, specifying either double
                or single precision floating points

        Returns:
            DisplacementField of size `(N, 2, H, W)`, or `(1, 2, H, W)` if
            `size` is given as an `int`

        If called on an instance of `torch.Tensor` or `DisplacementField`, the
        `size`, `device`, and `dtype` of that instance are used.
        For example

            df = DisplacementField(1,1,10,10)
            ident = df.identity_mapping()  # uses df.shape and df.device

        NOTE: If `use_identity_mapping_cache` is enabled, the returned field will
              be a reference to the field in cache. Use `clone()` on the returned
              field if you plan to perform inplace modifications and do not want
              to alter the cached version.
        """
        # find the right set of parameters
        params = DisplacementField._get_parameters(size, size, device, dtype)
        shape, batch_dim, device, tensor_type = [
            params[key] for key in ("shape", "batch_dim", "device", "tensor_type")
        ]

        # look in the cache and create from scratch if not there
        if (
            DisplacementField._cache_identities == True
            and (shape, device, tensor_type) in DisplacementField._identities
        ):
            Id = DisplacementField._identities[shape, device, tensor_type]
        else:
            id_theta = tensor_type([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], device=device)
            id_theta = id_theta.expand(batch_dim, *id_theta.shape[1:])
            Id = F.affine_grid(id_theta, shape, align_corners=False)
            Id = Id.permute(0, 3, 1, 2).field_()  # move the components to 2nd position

            if DisplacementField._cache_identities == True:
                DisplacementField._identities[shape, device, tensor_type] = Id

        return Id

    @classmethod
    def affine_field(cls, aff, size, offset=(0.0, 0.0), device=None, dtype=None):
        """Returns a displacement field for an affine transform within a bbox

        Args:
            aff: 2x3 ndarray or torch.Tensor. The affine matrix defining the
                affine transform
            offset: tuple with (x-offset, y-offset)
            size: an `int`, a `tuple` or a `torch.Size` of the form
                `(N, C, H, W)`. `C` is ignored.

        Returns:
            DisplacementField for the given affine transform of size
            `(N, 2, H, W)`, or `(1, 2, H, W)` if `size` is given as an `int`

        Note:
            the affine matrix defines the transformation that warps the
            destination to the source, such that,
            ```
            \vec{x_s} = A \vec{x_d}
            ```
            where x_s is a point in the source image, x_d a point in the
            destination image, and A is the affine matrix. The field returned
            will be defined over the destination image. So the matrix A should
            define the location in the source image that contribute to a pixel
            in the destination image.
        """
        params = DisplacementField._get_parameters(
            aff, size, device, dtype, override=True
        )
        device, dtype, tensor_type, size, batch_dim = [
            params[key]
            for key in ("device", "dtype", "tensor_type", "shape", "batch_dim")
        ]
        if isinstance(aff, list):
            aff = tensor_type(aff, device=device)
        if aff.ndimension() == 2:
            aff.unsqueeze_(0)
            N = 1
        elif aff.ndimension() == 3:
            N = aff.shape[0]
        else:
            raise ValueError(
                "Expected 2 or 3-dimensional affine matrix. "
                "Received shape {}.".format(aff.shape)
            )
        if N == 1 and batch_dim > 1:
            aff = aff.expand(batch_dim, *aff.shape[1:])
            N = batch_dim
        if offset[0] != 0 or offset[1] != 0:
            z = tensor_type([[0.0, 0.0, 1.0]], device=device)
            z = z.expand(N, *z.shape)
            A = torch.cat([aff, z], 1)
            B = tensor_type(
                [[1.0, 0.0, offset[0]], [0.0, 1.0, offset[1]], [0.0, 0.0, 1.0]],
                device=device,
            )
            B = B.expand(N, *B.shape)
            Bi = tensor_type(
                [[1.0, 0.0, -offset[0]], [0.0, 1.0, -offset[1]], [0.0, 0.0, 1.0]],
                device=device,
            )
            Bi = Bi.expand(N, *Bi.shape)
            aff = torch.mm(Bi, torch.mm(A, B))[:, :2]
        M = F.affine_grid(aff, size, align_corners=False)
        # Id is an identity mapping without the overhead of `identity_mapping`
        id_aff = tensor_type([[1, 0, 0], [0, 1, 0]], device=device)
        id_aff = id_aff.expand(N, *id_aff.shape)
        Id = F.affine_grid(id_aff, size, align_corners=False)
        M = M - Id
        M = M.permute(0, 3, 1, 2).field_()  # move the components to 2nd position
        return M

    # Basic vector field properties

    def is_identity(self, eps=None, magn_eps=None):
        """Checks if this is the identity displacement field, up to some
        tolerance `eps`, which is 0 by default.

        Args:
            eps: can either be a floating point number or a tensor of the same
                shape, in which case each location in the field can have a
                different tolerance.
            magn_eps: similar to eps, except bounds the magnitude of each
                vector instead of the components.

        If neither `eps` nor `magn_eps` are specified, the default is zero
        tolerance.

        Note that this does NOT check for identity mappings created by
        `identity_mapping()`. To check for that, subtract
        `self.identity_mapping()` first.

        This function is called and negated by `__bool__()`, which makes
        the following equivalent:

            if df:
                do_something()

        and

            if not df.is_identity():
                do_something()

        since `df.is_identity()` is equivalent to `not df`.
        """
        if eps is None and magn_eps is None:
            return (self == 0.0).all()
        else:
            is_id = True
            if eps is not None:
                is_id = is_id and (self >= -eps).all() and (self <= eps).all()
            if magn_eps is not None:
                is_id = is_id and (self.magnitude(True) <= magn_eps).all()
            return is_id

    def __bool__(self):
        return not self.is_identity().tensor_()

    __nonzero__ = __bool__

    def magnitude(self, keepdim=False):
        """Computes the magnitude of the displacement at each location in the
        displacement field

        Args:
            self: `DisplacementField` of shape `(N, 2, H, W)`

        Returns:
            `torch.Tensor` of shape `(N, H, W)` or `(N, 1, H, W)` if
            `keepdim` is `True`, containing the magnitude of the displacement
        """
        return self.tensor().pow(2).sum(dim=-3, keepdim=keepdim).sqrt()

    def distance(self, other, keepdim=False) -> torch.Tensor:
        """Compute the pointwise Euclidean distance between two displacement
        fields

        Args:
            self, other: DisplacementFields of the same shape `(N, 2, H, W)`

        Returns:
            `torch.Tensor` of shape `(N, H, W)` or `(N, 1, H, W)` if
            `keepdim` is `True`, containing the distance at each location in
            the displacement fields
        """
        return (self - other).magnitude(keepdim=keepdim)

    def mean_vector(self, keepdim=False):
        """Compute the mean displacement vector of each field in a batch

        Args:
            self: DisplacementFields of shape `(N, 2, H, W)`
            keepdim: if `True`, retains the spatial dimensions in the output

        Returns:
            `torch.Tensor` of shape `(N, 2)` or `DisplacementField` of shape
            `(N, 2, 1, 1)` if `keepdim` is `True`, containing the mean vector
            of each field
        """
        if keepdim:
            return self.mean(-1, keepdim=keepdim).mean(-2, keepdim=keepdim)
        else:
            return self.mean(-1).mean(-1)

    def mean_finite_vector(self, keepdim=False):
        """Compute the mean displacement vector of the finite elements in
        each field in a batch

        Args:
            self: DisplacementFields of shape `(N, 2, H, W)`
            keepdim: if `True`, retains the spatial dimensions in the output

        Returns:
            `torch.Tensor` of shape `(N, 2)` or `DisplacementField` of shape
            `(N, 2, 1, 1)` if `keepdim` is `True`, containing the mean finite
            vector of each field
        """
        mask = torch.isfinite(self).all(-3, keepdim=True)
        self = self.where(mask, torch.tensor(0).to(self))
        if keepdim:
            sum = self.sum(-1, keepdim=keepdim).sum(-2, keepdim=keepdim)
            count = mask.sum(-1, keepdim=keepdim).sum(-2, keepdim=keepdim)
        else:
            sum = self.sum(-1).sum(-1)
            count = mask.sum(-1).sum(-1)
        return sum / count.clamp(min=1).float()

    def mean_nonzero_vector(self, keepdim=False):
        """Compute the mean displacement vector of the nonzero elements in
        each field in a batch

        Note: to get the mean displacement vector of all elements, run

            field.mean(-1).mean(-1)

        Args:
            self: DisplacementFields of shape `(N, 2, H, W)`
            keepdim: if `True`, retains the spatial dimensions in the output

        Returns:
            `torch.Tensor` of shape `(N, 2)` or `DisplacementField` of shape
            `(N, 2, 1, 1)` if `keepdim` is `True`, containing the mean nonzero
            vector of each field
        """
        mask = self.magnitude(keepdim=True) > 0
        if keepdim:
            sum = self.sum(-1, keepdim=keepdim).sum(-2, keepdim=keepdim)
            count = mask.sum(-1, keepdim=keepdim).sum(-2, keepdim=keepdim)
        else:
            sum = self.sum(-1).sum(-1)
            count = mask.sum(-1).sum(-1)
        return sum / count.clamp(min=1).float()

    def min_vector(self, keepdim=False):
        """Compute the minimum displacement vector of each field in a batch

        Args:
            self: DisplacementFields of shape `(N, 2, H, W)`
            keepdim: if `True`, retains the spatial dimensions in the output

        Returns:
            `torch.Tensor` of shape `(N, 2)` or `DisplacementField` of shape
            `(N, 2, 1, 1)` if `keepdim` is `True`, containing the minimum
            vector of each field
        """
        if keepdim:
            return self.min(-1, keepdim=keepdim).values.min(-2, keepdim=keepdim).values
        else:
            return self.min(-1).values.min(-1).values

    def max_vector(self, keepdim=False):
        """Compute the maximum displacement vector of each field in a batch

        Args:
            self: DisplacementFields of shape `(N, 2, H, W)`
            keepdim: if `True`, retains the spatial dimensions in the output

        Returns:
            `torch.Tensor` of shape `(N, 2)` or `DisplacementField` of shape
            `(N, 2, 1, 1)` if `keepdim` is `True`, containing the maximum
            vector of each field
        """
        if keepdim:
            return self.max(-1, keepdim=keepdim).values.max(-2, keepdim=keepdim).values
        else:
            return self.max(-1).values.max(-1).values

    # Conversions to and from other representations of the displacement field

    def pixels(self, size=None):
        """Convert the displacement distances to units of pixels from the
        standard [-1, 1] distance convention.

        Note that while out of convenience, the type of
        the result is `DisplacementField`, many `DisplacementField`
        operations on it will produce incorrect results, since it will
        be in the wrong units.

        Args:
            self (DisplacementField): the field to convert
            size (int or torch.Size): the size, in pixels, of the tensor to be
                sampled. Used to calculate the pixel size. If not specified
                the size is assumed to be the size of the displacement field.

        Returns:
            a `DisplacementField` type tensor containing displacements in
            units of pixels
        """
        if size is None:
            size = self.shape
        if isinstance(size, tuple):
            size = size[-1]
        return self * (size / 2)

    def from_pixels(self, size=None):
        """Convert the displacement distances from units of pixels to the
        standard [-1, 1] distance convention.

        This reverses the operation of `pixels()`

        Args:
            self (DisplacementField): the field to convert
            size (int or torch.Size): the size, in pixels, of the tensor to be
                sampled. Used to calculate the pixel size. If not specified
                the size is assumed to be the size of the displacement field.

        Returns:
            a `DisplacementField` type tensor containing displacements in
            units of pixels
        """
        if size is None:
            size = self.shape
        if isinstance(size, tuple):
            size = size[-1]
        return self / (size / 2)

    def mapping(self):
        """Convert the displacement field to a mapping, where each location
        contains the coordinates of another location to which it maps.

        Note that while out of convenience, the type of
        the result is `DisplacementField`, many `DisplacementField`
        operations on it will produce incorrect results, since it will
        be in the wrong units.

        The units of the mapping will be in the standard [-1, 1] convention.

        Args:
            self (DisplacementField): the field to convert

        Returns:
            a `DisplacementField` type tensor containing the same field
            represented as a mapping
        """
        return self + self.identity_mapping()

    def from_mapping(self):
        """Convert a mapping to a displacement field which contains the
        displacement at each location.

        The units of the mapping should be in the standard [-1, 1] convention.

        Args:
            self (DisplacementField): the mapping to convert

        Returns:
            a `DisplacementField` containing the mapping represented
            as a displacement field
        """
        return self - self.identity_mapping()

    def pixel_mapping(self, size=None):
        """Convert the displacement field to a pixel mapping, where each pixel
        contains the coordinates of another pixel to which it maps.

        Note that while out of convenience, the type of
        the result is `DisplacementField`, many `DisplacementField`
        operations on it will produce incorrect results, since it will
        be in the wrong units.

        The units of the mapping will be in pixels in the range [0, size-1].

        Args:
            self (DisplacementField): the field to convert
            size (int or torch.Size): the size, in pixels, of the tensor to be
                sampled. Used to calculate the pixel size. If not specified
                the size is assumed to be the size of the displacement field.

        Returns:
            a `DisplacementField` type tensor containing the same field
            represented as a pixel mapping
        """
        if size is None:
            size = self.shape
        if isinstance(size, tuple):
            size = size[-1]
        return self.mapping().pixels(size) + (size - 1) / 2

    def from_pixel_mapping(self, size=None):
        """Convert a mapping to a displacement field which contains the
        displacement at each location.

        The units of the mapping should be in pixels in the range [0, size-1].

        Args:
            self (DisplacementField): the pixel mapping to convert
            size (int or torch.Size): the size, in pixels, of the tensor to be
                sampled. Used to calculate the pixel size. If not specified
                the size is assumed to be the size of the displacement field.

        Returns:
            a `DisplacementField` containing the pixel mapping represented
            as a displacement field
        """
        if size is None:
            size = self.shape
        if isinstance(size, tuple):
            size = size[-1]
        return (self - (size - 1) / 2).from_pixels(size).from_mapping()

    # Aliases for the components of the displacent vectors

    @property
    def x(self):
        """The column component of the displacent field
        """
        return self[..., 0:1, :, :]

    @x.setter
    def x(self, value):
        self[..., 0:1, :, :] = value

    j = x  # j & x are both aliases for the column component of the displacent

    @property
    def y(self):
        """The row component of the displacent field
        """
        return self[..., 1:2, :, :]

    @y.setter
    def y(self, value):
        self[..., 1:2, :, :] = value

    i = y  # i & y are both aliases for the row component of the displacent

    # Functions for sampling, composing, mapping, warping

    @ensure_dimensions(ndimensions=4, arg_indices=(1, 0), reverse=True)
    def sample(self, input, mode="bilinear", padding_mode="zeros"):
        r"""A wrapper for the PyTorch grid sampler to sample or warp and image
        by a displacent field.

        The displacement vector field encodes relative displacements from
        which to pull from the input, where vectors with values -1 or +1
        reference a displacement equal to the distance from the center point
        to the edges of the input.

        Args:
            `input` (Tensor): should be a PyTorch Tensor or DisplacementField
                on the same GPU or CPU as `self`, with `input` having
                dimensions :math:`(N, C, H_in, W_in)`, whenever `self` has
                dimensions :math:`(N, 2, H_out, W_out)`.
                The shape of the output will be :math:`(N, C, H_out, W_out)`.
            `mode` (str): 'bilinear' or 'nearest'
            `padding_mode` (str): determines the value sampled when a
                displacement vector's source falls outside of the input.
                Options are:
                 * "zeros" : produce the value zero (okay for sampling images
                            with zero as background, but potentially
                            problematic for sampling masks and terrible for
                            sampling from other displacement vector fields)
                 * "border" : produces the value at the nearest inbounds pixel
                              (great for sampling from masks and from other
                              residual displacement fields)
                 * "reflection" : reflects any sampling points that lie out
                                  of bounds until they fall inside the
                                  sampling range

        Returns:
            `output` (Tensor): the input after being warped by `self`,
            having shape :math:`(N, C, H_out, W_out)`

        See the PyTorch documentation of the underlying function for additional
        details:
        https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.grid_sample
        but note that the conventions used there are different.
        """
        field = self + self.identity_mapping()
        field = field.permute(0, 2, 3, 1)  # move components to last position
        out = F.grid_sample(
            input, field, mode=mode, padding_mode=padding_mode, align_corners=False
        )
        if not isinstance(input, DisplacementField):
            out.tensor_()
        return out

    def compose_with(self, other, mode="bilinear"):
        r"""Compose this displacement field with another displacement field.
        If `f = self` and `g = other`, then this computes
        `f⚬g` such that `(f⚬g)(x) ~= f(g(x))` for any tensor `x`.

        Returns:
            a displacement field such that when it is used to sample a tensor,
            it is the (approximate) equivalent of sampling with `other`
            and then with `self`.

        The reason this is only an approximate equivalence is because when
        sampling twice, information is inevitably lost in the intermediate
        stage. Sampling with the composed field is therefore more precise.
        """
        return self + self.sample(other, padding_mode="border")

    def __call__(self, x, mode="bilinear"):
        """Syntactic sugar for `compose_with()` or `sample()`, depending on
        the type of the sampled tensor.

        Be careful when using this that the sampled tensor is of the correct
        type for the desired outcome.
        For better assurance, it can be safer to call the functions explicitly.
        """
        if isinstance(x, DisplacementField):
            return self.compose_with(x, mode=mode)
        else:
            return self.sample(x, mode=mode)

    def multicompose(self, *others):
        """Composes multiple displacement fields with one another.
        This takes a list of fields :math:`f_0, f_1, ..., f_n`
        and composes them to get
        :math:`f_0 ⚬ f_1 ⚬ ... ⚬ f_n ~= f_0(f_1(...(f_n)))`

        Use of this function is not always recommended because of the
        potential for boundary effects when composing multiple displacements.
        Specifically, whenever a vector samples from out of bounds, the
        nearest vector is used, which may not be the desired behavior and can
        become a worse approximation of it as more displacement fields are
        composed together.
        """
        f = self
        for g in others:
            f = (f)(g)
        return f

    @ensure_dimensions(ndimensions=4, arg_indices=(0), reverse=True)
    def up(self, mips=None, scale_factor=2):
        """Upsamples by `mips` mip levels or by a factor of `scale_factor`,
        whichever one is specified.
        If neither are specified explicitly, upsamples by a factor of two, or
        in other words, one mip level.
        """
        if mips is not None:
            scale_factor = 2 ** mips
        if scale_factor == 1:
            return self
        return F.interpolate(
            self, scale_factor=scale_factor, mode="bilinear", align_corners=False
        )

    @ensure_dimensions(ndimensions=4, arg_indices=(0), reverse=True)
    def down(self, mips=None, scale_factor=2):
        """Downsample by `mips` mip levels or by a factor of `scale_factor`,
        whichever one is specified.
        If neither are specified explicitly, downsamples by a factor of two, or
        in other words, one mip level.
        """
        if mips is not None:
            scale_factor = 2 ** mips
        if scale_factor == 1:
            return self
        return F.interpolate(
            self, scale_factor=1.0 / scale_factor, mode="bilinear", align_corners=False
        )

    # Displacement Field Inverses

    def inverse(self, *args, **kwargs):
        """Return a symmetric inverse approximation for the displacement field

        Given a displacement field `f`, its symmetric inverse is a displacement
        field `f_inv` such that
        `f(f_inv) ~= identity ~= f_inv(f)`

        In other words
        :math:`f_{inv} = \argmin_{g} |f(g)|^2 + |g(f)|^2`

        Note that this is an approximation for the symmetric inverse.
        In cases for which only one inverse direction is desired, a better
        one-sided approximation can be achieved using `linverse()` or
        `rinverse()`.

        Also note that this overrides the `inverse()` method of `torch.Tensor`,
        but this definition cannot conflict, since `torch.Tensor.inverse` is
        only able to accept 2-dimensional tensors, and a `DisplacementField`
        is always at least 3-dimensional (2 spatial + 1 component dimension).
        """
        # TODO: Implement symmetric inverse. Currently using left inverse.
        return self.linverse(*args, **kwargs)

    def __invert__(self, *args, **kwargs):
        """Return a symmetric inverse approximation for the displacement field

        Given a displacement field `f`, its symmetric inverse is a displacement
        field `f_inv` such that
        `f(f_inv) ~= identity ~= f_inv(f)`

        In other words
        :math:`f_{inv} = \argmin_{g} |f(g)|^2 + |g(f)|^2`

        Note that this is an approximation for the symmetric inverse.
        In cases for which only one inverse direction is desired, a better
        approximation can be achieved using `linverse()` and `rinverse()`.

        This is syntactic sugar for `inverse()`, and allows the symmetric
        inverse to be called as `~f` rather than `f.inverse()`.
        """
        return self.inverse(*args, **kwargs)

    @wraps(inversion.linverse)
    def linverse(self, autopad=True):
        return inversion.linverse(self, autopad=True)

    @wraps(inversion.rinverse)
    def rinverse(self, *args, **kwargs):
        return inversion.rinverse(self, autopad=True)

    # Adapting functions inherited from torch.Tensor

    @permute_output
    @permute_input
    def fft(self, *args, **kwargs):
        return super(type(self), self).fft(*args, **kwargs)

    @permute_output
    @permute_input
    def ifft(self, *args, **kwargs):
        return super(type(self), self).ifft(*args, **kwargs)

    @permute_output
    def rfft(self, *args, **kwargs):
        # Present for completeness, but cannot be called on a DisplacementField
        return super(type(self), self).rfft(*args, **kwargs)

    @permute_input
    def irfft(self, *args, **kwargs):
        return super(type(self), self).irfft(*args, **kwargs)

    def __rpow__(self, other):
        # defined explicitly since pytorch default gives infinite recursion
        return self.new_tensor(other).__pow__(self)

    # Vector Voting

    @wraps(voting.gaussian_blur)
    def gaussian_blur(self, sigma=1, kernel_size=5):
        return voting.gaussian_blur(self, sigma, kernel_size)

    @wraps(voting.vote)
    def get_vote_shape(self):
        return voting.get_vote_shape(self)

    @wraps(voting.vote)
    def get_subset_size(self, subset_size=None):
        return voting.get_subset_size(self, subset_size)

    @wraps(voting.vote)
    def get_vote_subsets(self, subset_size=None):
        return voting.get_vote_subsets(self, subset_size)

    @wraps(voting.vote)
    def linear_combination(self, weights):
        return voting.linear_combination(self, weights=weights)

    @wraps(voting.vote)
    def smoothed_combination(self, weights, blur_sigma=2., kernel_size=5):
        return voting.smoothed_combination(self, 
                                           weights=weights, 
                                           blur_sigma=blur_sigma, 
                                           kernel_size=kernel_size)

    @wraps(voting.vote)
    def get_vote_weights(self, softmin_temp=1, blur_sigma=1, subset_size=None):
        return voting.get_vote_weights(self, softmin_temp, blur_sigma, subset_size)

    @wraps(voting.vote)
    def vote(self, softmin_temp=1, blur_sigma=1, subset_size=None):
        return voting.vote(self, softmin_temp, blur_sigma, subset_size)

    @wraps(voting.vote)
    def get_vote_weights_with_variances(
        self, var, softmin_temp=1, blur_sigma=1, subset_size=None
    ):
        return voting.get_vote_weights_with_variances(
            self, var, softmin_temp, blur_sigma, subset_size
        )

    @wraps(voting.vote)
    def vote_with_variances(self, var, softmin_temp=1, blur_sigma=1, subset_size=None):
        return voting.vote_with_variances(
            self, var, softmin_temp, blur_sigma, subset_size
        )

    @wraps(voting.vote)
    def get_vote_weights_with_distances(
        self, distances, softmin_temp=1, blur_sigma=1, subset_size=None
    ):
        return voting.get_vote_weights_with_distances(
            self, distances, softmin_temp, blur_sigma, subset_size
        )

    @wraps(voting.vote)
    def vote_with_distances(
        self, distances, softmin_temp=1, blur_sigma=1, subset_size=None
    ):
        return voting.vote_with_distances(
            self, distances, softmin_temp, blur_sigma, subset_size
        )

    @wraps(voting.vote)
    def get_priority_vote_weights(
        self, priorities, consensus_threshold=2, subset_size=None
    ):
        return voting.get_priority_vote_weights(
            self,
            priorities,
            consensus_threshold=consensus_threshold,
            subset_size=subset_size,
        )

    @wraps(voting.vote)
    def priority_vote(
        self, priorities, consensus_threshold=2, blur_sigma=1, subset_size=None
    ):
        return voting.priority_vote(
            self,
            priorities,
            consensus_threshold=consensus_threshold,
            blur_sigma=blur_sigma,
            subset_size=subset_size,
        )

class set_identity_mapping_cache():
    """Context-manager that controls caching of identity_mapping() results.

    ``set_identity_mapping_cache`` will enable or disable the cache (:attr: `mode`).
    It can be used as a context-manager or as a function.

    If enabled, cache results of identity_mapping() calls based on
    (shape, device, dtype) for faster recall.

    This may also improve repeated calls to other torchfields methods,
    such as `sample()`, `rand_in_bounds()`, `pixel_mapping()`, etc.

    The trade-off is a higher burden on CPU/GPU memory, therefore
    caching is disabled by default.

    Args:
      mode (bool): Flag whether to enable cache (``True``), or disable (``False``).
      clear_cache (bool): Optional flag whether or not to empty any existing
        cached results. Default is ``False``.

    Note: For performance reasons, the returned field from identity_mapping()
      will be the cached, *mutable* field. Use `clone()` on the returned field
      if you plan to perform in-place modifications and do not want to alter the
      cached version.
    """

    def __init__(self, mode: bool, clear_cache: bool = False) -> None:
        self.prev = DisplacementField.is_identity_mapping_cache_enabled()
        DisplacementField._set_identity_mapping_cache(mode, clear_cache)

    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        DisplacementField._set_identity_mapping_cache(self.prev)
