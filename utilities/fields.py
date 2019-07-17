"""PyTorch tensor type for working with displacement vector fields
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations
from functools import wraps
from utilities.helpers import GaussianSmoothing


#############################################
# Decorators for enforcing return value types
#############################################


def return_subclass_type(cls):
    """Class decorator for a subclass to encourage it to return its own
    subclass type whenever its inherited functions would otherwise return
    the superclass type.

    This works by attempting to convert any return values of the superclass
    type to the subclass type, and then defaulting back to the original
    return value on any errors during conversion.

    If running the subclass constructor has undesired side effects,
    the class can define a `_from_superclass()` function that casts
    to the subclass type more directly.
    This function should raise an exception if the type is not compatible.
    If `_from_superclass` is not defined, the class constructor is called
    by default.
    """
    def decorator(f):
        @wraps(f)
        def f_decorated(*args, **kwargs):
            out = f(*args, **kwargs)
            try:
                if not isinstance(out, cls) and isinstance(out, cls.__bases__):
                    return cls._from_superclass(out)
            except Exception:
                pass
            # Result cannot be returned as subclass type
            return out
        return f_decorated

    # fall back to constructor if _from_superclass not defined
    try:
        cls._from_superclass
    except AttributeError:
        cls._from_superclass = cls

    for name in dir(cls):
        attr = getattr(cls, name)
        if name not in dir(object) and callable(attr):
            try:
                # check if this attribute is flagged to keep its return type
                if attr._keep_type:
                    continue
            except AttributeError:
                pass
            setattr(cls, name, decorator(attr))
    return cls


def dec_keep_type(keep=True):
    """Function decorator that adds a flag to tell `return_subclass_type()`
    to leave the function's return type as is.

    This is useful for functions that intentionally return a value of
    superclass type.

    If a boolean argument is passed to the decorator as

        @dec_keep_type(True)
        def func():
            pass

    then that agument determines whether to enable the flag. If no argument
    is passed, the flag is enabled as if `True` were passed.

        @dec_keep_type
        def func():
            pass

    """
    def _dec_keep_type(keep_type):
        def _set_flag(f):
            f._keep_type = keep_type
            return f
        return _set_flag
    if isinstance(keep, bool):  # boolean argument passed
        return _dec_keep_type(keep)
    else:  # the argument is actually the function itself
        func = keep
        return _dec_keep_type(True)(func)


####################################
# DisplacementField Class Definition
####################################


@return_subclass_type
class DisplacementField(torch.Tensor):
    """An abstraction that encapsulates functionality of displacement fields
    as used in Spatial Transformer Networks.

    DisplacementFields can be treated as normal PyTorch tensors for most
    purposes, and also include additional functionality for composing
    displacements and sampling from tensors.
    """

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, *args, **kwargs):
        if len(self.shape) < 3:
            raise ValueError('The displacement field must have a components '
                             'dimension. Only {} dimensions are present.'
                             .format(len(self.shape)))
        if self.shape[-3] != 2:
            raise ValueError('The displacement field must have exactly 2 '
                             'components, not {}.'.format(self.shape[-3]))

    def __repr__(self, *args, **kwargs):
        out = super().__repr__(*args, **kwargs)
        return out.replace('tensor', 'field', 1).replace('\n ', '\n')

    # Conversion to and from torch.Tensor

    @dec_keep_type
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
                "'{}' cannot be converted to '{}'. Valid options are: {}"
                .format(type(self).__name__, DisplacementField.__name__,
                        [base.__module__ + "." + base.__name__
                         for base in allowed_types]))
        if len(self.shape) < 3:
            raise ValueError('The displacement field must have a components '
                             'dimension. Only {} dimensions are present.'
                             .format(len(self.shape)))
        if self.shape[-3] != 2:
            raise ValueError('The displacement field must have exactly 2 '
                             'components, not {}.'.format(self.shape[-3]))
        self.__class__ = DisplacementField
        self.__init__(*args, **kwargs)  # in case future __init__ is nonempty
        return self
    torch.Tensor.field_ = field_  # adds conversion to torch.Tensor superclass
    _from_superclass = field_  # for use in `return_subclass_type()`

    @dec_keep_type
    def field(data, *args, **kwargs):
        """Converts a `torch.Tensor` to a `DisplacementField`
        """
        if isinstance(data, torch.Tensor):
            return DisplacementField.field_(data.clone(), *args, **kwargs)
        else:
            return DisplacementField.field_(
                torch.tensor(data, *args, **kwargs).float())
    torch.Tensor.field = field  # adds conversion to torch.Tensor superclass
    torch.field = field

    @dec_keep_type
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

    @dec_keep_type
    def tensor(self):
        """Converts the `DisplacementField` to a standard `torch.Tensor`
        """
        return self.clone().tensor_()

    # Decorators to convert the inputs and outputs of DisplacementField methods

    def permute_input(f):
        """Function decorator to permute the input dimensions from the
        DisplacementField convention `(N, 2, H, W)` to the standard PyTorch
        field convention `(N, H, W, 2)` before passing it into the function.
        """
        @wraps(f)
        def f_new(self, *args, **kwargs):
            ndims = self.ndimension()
            perm = self.permute(*range(ndims-3), -2, -1, -3)
            return f(perm, *args, **kwargs)
        return f_new

    def permute_output(f):
        """Function decorator to permute the dimensions of the function output
        from the standard PyTorch field convention `(N, H, W, 2)` to the
        DisplacementField convention `(N, 2, H, W)` before returning it.
        """
        @wraps(f)
        def f_new(self, *args, **kwargs):
            out = f(self, *args, **kwargs)
            ndims = out.ndimension()
            return out.permute(*range(ndims-3), -1, -3, -2)
        return f_new

    def ensure_dimensions(ndimensions=4, arg_indices=(0,), reverse=False):
        """Function decorator to ensure that the the input has the
        approprate number of dimensions

        If it has too few dimensions, it pads the input with dummy dimensions.

        Args:
            ndimensions (int): number of dimensions to pad to
            arg_indices (int or List[int]): the indices of inputs to pad
                Note: Currently, this only works on arguments passed by
                position. Those inputs must be a torch.Tensor or
                DisplacementField.
            reverse (bool): if `True`, it then also removes the added dummy
                dimensions from the output, down to the number of dimensions
                of arg[arg_indices[0]]
        """
        if callable(ndimensions):  # it was called directly on a function
            func = ndimensions
            ndimensions = 4
        else:
            func = None
        if isinstance(arg_indices, int):
            arg_indices = (arg_indices,)
        assert(len(arg_indices) > 0)

        def decorator(f):
            @wraps(f)
            def f_decorated(*args, **kwargs):
                args = list(args)
                original_ndims = len(args[arg_indices[0]].shape)
                for i in arg_indices:
                    if i >= len(args):
                        continue
                    while args[i].ndimension() < ndimensions:
                        args[i] = args[i].unsqueeze(0)
                out = f(*args, **kwargs)
                while reverse and out.ndimension() > original_ndims:
                    new_out = out.squeeze(0)
                    if new_out.ndimension() == out.ndimension():
                        break  # no progress made; nothing left to squeeze
                    out = new_out
                return out
            return f_decorated

        if func is None:  # parameters were passed to the decorator
            return decorator
        else:  # the function itself was passed to the decorator
            return decorator(func)

    # Constuctors for typical displacent fields

    def identity(*args, **kwargs):
        """Returns an identity displacent field (containing all zero vectors).

        See :func:`torch.zeros`
        """
        if len(args) > 0 and isinstance(args[0], torch.Tensor):
            tensor_like, *args = args
            if 'device' not in kwargs or kwargs['device'] is None:
                kwargs['device'] = tensor_like.device
            if 'size' not in kwargs or kwargs['size'] is None:
                kwargs['size'] = tensor_like.shape
            if 'dtype' not in kwargs or kwargs['dtype'] is None:
                kwargs['dtype'] = tensor_like.dtype
        return torch.zeros(*args, **kwargs)
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
            if 'device' not in kwargs or kwargs['device'] is None:
                kwargs['device'] = tensor_like.device
            if 'size' not in kwargs or kwargs['size'] is None:
                kwargs['size'] = tensor_like.shape
            if 'dtype' not in kwargs or kwargs['dtype'] is None:
                kwargs['dtype'] = tensor_like.dtype
        return torch.ones(*args, **kwargs)
    ones_like = ones

    def rand(*args, **kwargs):
        """Returns a displacement field type tensor with each vector
        component randomly sampled from the uniform distribution on [0, 1).

        See :func:`torch.rand`
        """
        if len(args) > 0 and isinstance(args[0], torch.Tensor):
            tensor_like, *args = args
            if 'device' not in kwargs or kwargs['device'] is None:
                kwargs['device'] = tensor_like.device
            if 'size' not in kwargs or kwargs['size'] is None:
                kwargs['size'] = tensor_like.shape
            if 'dtype' not in kwargs or kwargs['dtype'] is None:
                kwargs['dtype'] = tensor_like.dtype
        return torch.rand(*args, **kwargs)
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

    def _get_parameters(tensor, size, device, dtype):
        """Auxiliary function to deduce the right set of parameters to a tensor
        function.
        In particular, if `tensor` is a `torch.Tensor`, it uses those values.
        Otherwise, if the values are not explicitly specified, returns the
        default values
        """
        if isinstance(tensor, torch.Tensor):
            size = tensor.shape
            device = tensor.device
            dtype = tensor.dtype
        else:
            if device is None:
                try:
                    device = torch.cuda.current_device()
                except AssertionError:
                    device = 'cpu'
            if dtype is None:
                dtype = torch.float
        if isinstance(size, tuple):
            orig_shape = size
            batch_dim = size[0] if len(size) > 3 else 1
            if size[-1] == size[-2]:
                size = size[-2]
            else:
                raise ValueError("Bad size: {}. Expected a square tensor size."
                                 .format(size))
        else:
            try:
                orig_shape = torch.Size((1, 2, size, size))
            except TypeError:
                raise TypeError("'size' must be an 'int', 'tuple', or "
                                "'torch.Size'. Received '{}'"
                                .format(type(size).__qualname__))
            batch_dim = 1
        tensor_types = {
            (True, torch.double): torch.DoubleTensor,
            (True, torch.float): torch.FloatTensor,
            (False, torch.double): torch.cuda.DoubleTensor,
            (False, torch.float): torch.cuda.FloatTensor,
        }
        return {
            'shape': orig_shape,
            'size': size,
            'batch_dim': batch_dim,
            'device': device,
            'dtype': dtype,
            'tensor_type': tensor_types['cpu' in str(device), dtype]
        }

    @torch.no_grad()
    def identity_mapping(size, device=None, dtype=None, cache=None):
        """Returns an identity mapping with -1 and +1 at the corners of the
        image (not the centers of the border pixels as in PyTorch 1.1).

        Note that this is NOT an identity displacement field, and therefore
        sampling with it will not return the input.
        To get the identity displacement field, use `identity()`.
        Instead, this creates a mapping that maps each coordinate to its
        own coordinate vector (in the [-1, +1] space).

        Args:
            size: either an `int` or a `torch.Size` of the form `(N, C, H, W)`.
                `H` and `W` must be the same (a square tensor).
                `N` and `C` are ignored.
            cache (bool): Use `cache = True` to cache the identity of this
                size for faster recall. This speed-up can be a burden on
                cpu/gpu memory, however, so it is disabled by default.
            device (torch.device): the device (cpu/cuda) on which to create
                the mapping
            dtype (torch.dtype): the data type of resulting mapping. Can be
                `torch.float` or `torch.double`, specifying either double
                or single precision floating points

        If called on an instance of `torch.Tensor` or `DisplacementField`, the
        `size`, `device`, and `dtype` of that instance are used.
        For example

            df = DisplacementField(1,1,10,10)
            ident = df.identity_mapping()  # uses df.shape and df.device
        """
        def _create_identity_mapping(size, device, tensor_type):
            id_theta = tensor_type([[[1, 0, 0], [0, 1, 0]]], device=device)
            Id = F.affine_grid(id_theta, torch.Size((1, 1, size, size)))
            Id *= (size - 1) / size  # rescale the identity provided by PyTorch
            Id = Id.permute(0, 3, 1, 2)  # move the components to 2nd position
            return Id
        # find the right set of parameters
        params = DisplacementField._get_parameters(size, size, device, dtype)
        orig_shape, batch_dim, size, device, dtype, tensor_type = \
            [params[key] for key in ('shape', 'batch_dim', 'size',
                                     'device', 'dtype', 'tensor_type')]
        # initialize the cache if this is the first call
        try:
            DisplacementField._identities
        except AttributeError:
            DisplacementField._identities = {}
        # look in the cache and create from scratch if not there
        if (size, dtype) in DisplacementField._identities:
            Id = DisplacementField._identities[size, dtype].copy()
        else:
            Id = _create_identity_mapping(size, device, tensor_type)
            if cache:
                DisplacementField._identities[size, dtype] = Id.copy()
        # reshape to the desired dimensions and move to the desired device
        if batch_dim > 1:
            Id = Id.expand(batch_dim, *orig_shape[1:])
        else:
            Id = Id.view(orig_shape)
        return Id.to(device=device, dtype=dtype)

    @classmethod
    def affine_field(cls, aff, size, offset=(0., 0.), device=None, dtype=None):
        """Returns a displacement field for an affine transform within a bbox

        Args:
            aff: 2x3 ndarray or torch.Tensor. The affine matrix defining the 
                affine transform
            offset: tuple with (x-offset, y-offset)
            size: an `int`, a `tuple` or a `torch.Size` of the form
                `(N, C, H, W)`. `H` and `W` must be the same (a square tensor).
                `N` and `C` are ignored.

        Returns:
            DisplacementField for the given affine transform

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
        params = DisplacementField._get_parameters(aff, size, device, dtype)
        device, dtype, tensor_type = \
            [params[key] for key in ('device', 'dtype', 'tensor_type')]
        if not isinstance(size, tuple):
            try:
                size = torch.Size((1, 1, size, size))
            except TypeError:
                raise TypeError("'size' must be an 'int', 'tuple', or "
                                "'torch.Size'. Received '{}'"
                                .format(type(size).__qualname__))
        if isinstance(aff, list):
            A = tensor_type(aff + [[0, 0, 1]], device=device)
        else:
            A = torch.cat([aff, tensor_type([[0, 0, 1]], device=device)])
        B = tensor_type([[1., 0, offset[0]],
                         [0, 1., offset[1]],
                         [0, 0, 1]], device=device)
        Bi = tensor_type([[1., 0, -offset[0]],
                          [0, 1., -offset[1]],
                          [0, 0, 1]], device=device)
        theta = torch.mm(Bi, torch.mm(A, B))[:2].unsqueeze(0)
        M = F.affine_grid(theta, size)
        # Id is an identity mapping without the overhead of `identity_mapping`
        id_theta = tensor_type([[[1, 0, 0], [0, 1, 0]]], device=device)
        Id = F.affine_grid(id_theta, size)
        M -= Id
        M *= (size[-2] - 1) / size[-2]  # rescale the grid provided by PyTorch
        M = M.permute(0, 3, 1, 2)  # move the components to 2nd position
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
        """  # TODO: allow a bound on the magnitude (|f|^2 < gamma)
        if eps is None and magn_eps is None:
            return (self == 0.).all()
        else:
            is_id = True
            if eps is not None:
                is_id = is_id and (self >= -eps).all() and (self <= eps).all()
            if magn_eps is not None:
                is_id = is_id and self.magnitude(keepdim=True) <= magn_eps
            return is_id

    def __bool__(self):
        return not self.is_identity()
    __nonzero__ = __bool__

    @dec_keep_type
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

    @dec_keep_type
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
        else:
            sum = self.sum(-1).sum(-1)
        count = mask.sum(-1).sum(-1)
        if count == 0:
            return sum
        else:
            return sum / count.float()

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
        if keepdim:
            sum = self.sum(-1, keepdim=keepdim).sum(-2, keepdim=keepdim)
        else:
            sum = self.sum(-1).sum(-1)
        count = (self.magnitude() > 0).sum(-1).sum(-1)
        if count == 0:
            return sum
        else:
            return sum / count.float()

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
            return (self.min(-1, keepdim=keepdim).values
                        .min(-2, keepdim=keepdim).values)
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
            return (self.max(-1, keepdim=keepdim).values
                        .max(-2, keepdim=keepdim).values)
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
        return (self.mapping() + 1).pixels(size) - .5

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
        return ((self + .5).from_pixels(size) - 1).from_mapping()

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

    @dec_keep_type
    @ensure_dimensions(ndimensions=4, arg_indices=(1, 0), reverse=True)
    def sample(self, input, mode='bilinear', padding_mode='zeros'):
        r"""A wrapper for the PyTorch grid sampler that uses size-agnostic
        residual conventions.

        The displacement vector field encodes relative displacements from
        which to pull from the input, where vectors with values -1 or +1
        reference a displacement equal to the distance from the center point
        to the actual edges of the input (as opposed to the centers of the
        border pixels as in PyTorch 1.0).

        Args:
            `input` (Tensor): should be a PyTorch Tensor or DisplacementField
                on the same GPU or CPU as `self`, with `input` having
                dimensions :math:`(N, C, H_in, W_in)`, whenever `self` has
                dimensions :math:`(N, 2, H_out, W_out)`.
                The shape of the output will be :math:`(N, C, H_out, W_out)`.
            `padding_mode` (str): determines the value sampled when a
                displacement vector's source falls outside of the input.
                Options are:
                 - "zeros" : produce the value zero (okay for sampling images
                            with zero as background, but potentially
                            problematic for sampling masks and terrible for
                            sampling from other displacement vector fields)
                 - "border" : produces the value at the nearest inbounds pixel
                              (great for sampling from masks and from other
                              residual displacement fields)

        Returns:
            `output` (Tensor): the input after being warped by `self`,
            having shape :math:`(N, C, H_out, W_out)`
        """
        field = self + self.identity_mapping()
        shape = input.shape
        if shape[-1] != shape[-2]:
            raise NotImplementedError('Sampling from non-square tensors '
                                      'not yet implemented here.')
        scaled_field = field * (shape[-2] / (shape[-2] - 1))
        scaled_field = scaled_field.permute(0, 2, 3, 1)
        out = F.grid_sample(input, scaled_field, mode=mode,
                            padding_mode=padding_mode)
        if isinstance(input, DisplacementField):
            out = DisplacementField._from_superclass(out)
        return out

    def compose_with(self, other, mode='bilinear'):
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
        return self + self.sample(other, padding_mode='border')

    @dec_keep_type
    def __call__(self, x, mode='bilinear'):
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
            scale_factor = 2**mips
        if scale_factor == 1:
            return self
        return F.interpolate(self, scale_factor=scale_factor,
                             mode='bilinear', align_corners=False)

    @ensure_dimensions(ndimensions=4, arg_indices=(0), reverse=True)
    def down(self, mips=None, scale_factor=2):
        """Downsample by `mips` mip levels or by a factor of `scale_factor`,
        whichever one is specified.
        If neither are specified explicitly, downsamples by a factor of two, or
        in other words, one mip level.
        """
        if mips is not None:
            scale_factor = 2**mips
        if scale_factor == 1:
            return self
        return F.interpolate(self, scale_factor=1./scale_factor,
                             mode='bilinear', align_corners=False)

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
        raise NotImplementedError

    def linverse(self, autopad=True):
        """Return a left inverse approximation for the displacement field

        Given a displacement field `f`, its left inverse is a displacement
        field `g` such that
        `g(f) ~= identity`

        In other words
        :math:`f_{inv} = \argmin_{g} |g(f)|^2`
        """
        # comparison to 0
        eps1 = 2**(-51) if self.dtype is torch.double else 2**(-23)
        # denominator fudge factor to avoid dividing by 0
        eps2 = eps1 * 2**(-10)
        # tolarance for point containment in a quadrilateral
        eps3 = 2**-16

        def tensor_min(*args):
            minimum, *rest = args
            for arg in rest:
                minimum = minimum.min(arg)
            return minimum

        def tensor_max(*args):
            maximum, *rest = args
            for arg in rest:
                maximum = maximum.max(arg)
            return maximum

        def fold(inp):
            """Collapse the matrix at each pixel onto the local neighborhood"""
            pad = (0, (inp.shape[2] + 1) % 2,  # last dimension
                   0, (inp.shape[1] + 1) % 2)  # second to last dimension
            res = F.pad(inp, pad)
            res = F.fold(
                res.view(1, res.shape[0]*res.shape[1]*res.shape[2], -1),
                output_size=inp.shape[3:], kernel_size=inp.shape[1:3],
                padding=((inp.shape[1])//2, (inp.shape[2])//2))
            return res

        def pad(inp):
            """Pads the field just enough to reduce border effects"""
            with torch.no_grad():
                *_, H, W = inp.shape
                mapping = inp.pixel_mapping()
                pad_yl = mapping.y[..., 0, :].max().ceil().int().item()
                pad_yh = (H-1-mapping.y[..., -1, :].min()).ceil().int().item()
                pad_xl = mapping.x[..., :, 0].max().ceil().int().item()
                pad_xh = (W-1-mapping.x[..., :, -1].min()).ceil().int().item()
                pad_yl = max(pad_yl, 0) + 1
                pad_yh = max(pad_yh, 0) + 1
                pad_xl = max(pad_xl, 0) + 1
                pad_xh = max(pad_xh, 0) + 1
                # ensure that the new field is square newH = newW
                newH, newW = pad_yl + H + pad_yh, pad_xl + W + pad_xh
                if newH > newW:
                    pad_xh += newH - newW
                elif newW > newH:
                    pad_yh += newW - newH
                pad = (pad_xl, pad_xh, pad_yl, pad_yh)
            return (F.pad(inp.pixels(), pad, mode='replicate').field()
                    .from_pixels(), pad)

        def unpad(inp, pad):
            """Crops the field back to the original size"""
            p_xl, p_xh, p_yl, p_yh = pad
            p_xh = inp.shape[-1] - p_xh
            p_yh = inp.shape[-2] - p_yh
            return inp.pixels()[..., p_yl:p_yh, p_xl:p_xh].from_pixels()

        @torch.no_grad()
        def winding_number(Px, Py, v00, v01, v10, v11):
            """Gives the winding number of a quadrilateral around a grid point
            When the winding number is non-zero, the quadrilateral contains
            the grid point. More specifically, only positive winding numbers
            are relevant in this context, since inverted quadrilaterals are
            not considered.
            For edge cases, this is a more accurate measure than checking
            whether i and j fall within the range [0,1).
            Based on http://geomalgorithms.com/a03-_inclusion.html
            """
            try:
                # vertices in counterclockwise order (viewing y as up)
                V = torch.stack([v00, v01, v11, v10, v00], dim=0).field_()
                v00 = v01 = v10 = v11 = None  # del v00, v01, v10, v11
                # initial and final vertex for each edge
                Vi, Vf = V[:-1], V[1:]
                V = None  # del V
                # sign of cross product indicates direction around grid point
                cross = ((Vf.x - Vi.x)*(Py - Vi.y) - (Px - Vi.x)*(Vf.y - Vi.y))
                # upward crossing of rightward ray from grid point
                upward = (Vi.y <= Py) & (Vf.y > Py) & (cross > -eps3)
                # downward crossing of rightward ray from grid point
                downward = (Vi.y > Py) & (Vf.y <= Py) & (cross < -eps3)
                Vi = Vf = Px = Py = cross = None  # del Vi, Vf, Px, Py, cross
                # winding number = diff between number of up and down crossings
                return (upward.int() - downward.int()).sum(dim=0)
            except RuntimeError:
                # In case this is an out-of-memory error, clear temp tensors
                v00 = v01 = v10 = v11 = Px = Py = None
                V = Vi = Vf = cross = upward = downward = None
                raise

        try:
            # pad the field
            if autopad:
                field, padding = pad(self)
            else:
                field = self
                padding = None

            # vectors at the four corners of each pixel's quadrilateral
            mapping = field.pixel_mapping()
            v00 = mapping[..., :-1, :-1]
            v01 = mapping[..., :-1, 1:]
            v10 = mapping[..., 1:, :-1]
            v11 = mapping[..., 1:, 1:]
            mapping = None  # del mapping

            with torch.no_grad():
                # find each quadrilateral's (set of 4 vectors) span, in pixels
                v_min = tensor_min(v00, v01, v10, v11).floor()
                v_min.y.clamp_(0, field.shape[-2] - 1)
                v_min.x.clamp_(0, field.shape[-1] - 1)
                v_max = tensor_max(v00, v01, v10, v11).ceil()
                v_max.y.clamp_(0, field.shape[-2] - 1)
                v_max.x.clamp_(0, field.shape[-1] - 1)
                # d_x and d_y are the largest spans in x and y
                d = (v_max - v_min).max_vector().max(0)[0].int()
                v_max = None  # del v_max
                d_x, d_y = list(d.cpu().numpy())
                d = ((d//2).unsqueeze(-1).unsqueeze(-1)  # center of the span
                     .unsqueeze(-1).unsqueeze(-1)).to(v_min)
                v_min.y.clamp_(0, field.shape[-2] - 1 - d_y)
                v_min.x.clamp_(0, field.shape[-1] - 1 - d_x)
                # u is an identity pixel mapping of a d_y by d_x neighborhood
                u = field.identity().pixel_mapping()[..., :d_y, :d_x].round()
                ux = u.x.unsqueeze(-1).unsqueeze(-1)
                uy = u.y.unsqueeze(-1).unsqueeze(-1)
                u = None  # del u

            # subtract out v_min to bring all quadrilaterals near zero
            v00 = (v00 - v_min).unsqueeze(-4).unsqueeze(-4)
            v01 = (v01 - v_min).unsqueeze(-4).unsqueeze(-4)
            v10 = (v10 - v_min).unsqueeze(-4).unsqueeze(-4)
            v11 = (v11 - v_min).unsqueeze(-4).unsqueeze(-4)

            # quadratic coefficients in gridsample solution `a*j^2+b*j+c=0`
            a = ((v00.x - v01.x) * (v00.y - v01.y - v10.y + v11.y)
                 - (v00.x - v01.x - v10.x + v11.x) * (v00.y - v01.y))
            b = ((ux - v00.x) * (v00.y - v01.y - v10.y + v11.y)
                 + (v00.x - v01.x) * (-v00.y + v10.y)
                 - (-v00.x + v10.x) * (v00.y - v01.y)
                 - (v00.x - v01.x - v10.x + v11.x) * (uy - v00.y))
            c = (ux - v00.x)*(-v00.y + v10.y) - (-v00.x + v10.x)*(uy - v00.y)
            # quadratic formula solution (note positive root is always invalid)
            j_temp = ((b + (b.pow(2) - 4*a*c).clamp(min=eps2).sqrt()).abs()
                      / ((2*a).abs() + eps2))
            # corner case when a == 0 (reduces to `b*j + c = 0`)
            j_temp = j_temp.where(a.abs() > eps1, -c / b)
            a = b = c = None  # del a, b, c
            # get i from j_temp
            i = ((uy - v00.y + (v00.y - v01.y) * j_temp).abs()
                 / ((-v00.y + v10.y + (v00.y - v01.y - v10.y + v11.y) * j_temp)
                    .abs() + eps2))
            j_temp = None  # del j_temp
            # j has significantly smaller rounding error for near-trapezoids
            j = ((ux - v00.x + (v00.x - v10.x) * i).abs()
                 / ((-v00.x + v01.x + (v00.x - v10.x - v01.x + v11.x) * i)
                    .abs() + eps2))
            # winding_number > 0 means point is contained in the quadrilateral
            wn = winding_number(ux, uy, v00, v01, v10, v11)
            ux = uy = None  # del ux, uy
            v00 = v01 = v10 = v11 = None  # del v00, v01, v10, v11

            # negative of the bilinear interpolation to produce inverse vector
            v00 = field[..., :-1, :-1].unsqueeze(-3).unsqueeze(-3)
            v01 = field[..., :-1, 1:].unsqueeze(-3).unsqueeze(-3)
            v10 = field[..., 1:, :-1].unsqueeze(-3).unsqueeze(-3)
            v11 = field[..., 1:, 1:].unsqueeze(-3).unsqueeze(-3)
            inv = -((1-i)*(1-j)*v00 + (1-i)*j*v01 + i*(1-j)*v10 + i*j*v11)
            v00 = v01 = v10 = v11 = None  # del v00, v01, v10, v11

            # mask out inverse vectors at points outside the quadrilaterals
            mask = (wn > 0) & torch.isfinite(i) & torch.isfinite(j)
            i = j = wn = None  # del i, j, wn
            inv = inv.where(mask, torch.tensor(0.).to(inv))
            # append mask to keep track of how many contributions to each pixel
            inv = torch.cat((inv, mask.to(inv)), 1)
            mask = None

            # indices at which to place each inverse vector in a sparse tensor
            indices = ((v_min.unsqueeze(-3).unsqueeze(-3) + d)
                       .view(2, -1).flip(0).round().long())
            v_min = d = None
            inv = inv.view(3, d_y, d_x, -1).permute(3, 0, 1, 2)
            # construct sparse tensor and use `to_dense` to arrange vectors
            inv = torch.cuda.sparse.FloatTensor(
                indices, inv, (*field.shape[-2:], 3, d_y, d_x),
                device=inv.device)
            inv = inv.to_dense().permute(2, 3, 4, 0, 1)
            # fold the d_y by d_x neighborhoods by summing the overlaps
            inv = fold(inv)
            # divide each pixel by number of contributions to get an average
            inv = inv[:, :2] / inv[:, 2:].clamp(min=1.)

            # crop back to original shape
            if autopad:
                inv = unpad(inv.field(), padding)
            return inv
        except RuntimeError:
            # In case this is an out-of-memory error, clear temporary tensors
            self = field = pad = mapping = v_min = v_max = d = d_x = d_y = None
            u = ux = uy = v00 = v01 = v10 = v11 = wn = None
            a = b = c = j_temp = i = j = None
            mask = inv = indices = None
            raise

    def rinverse(self, *args, **kwargs):
        """Return a right inverse approximation for the displacement field

        Given a displacement field `f`, its right inverse is a displacement
        field `g` such that
        `f(g) ~= identity`

        In other words
        :math:`f_{inv} = \argmin_{g} |f(g)|^2`
        """
        raise NotImplementedError

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
        # TODO: Implement symmetric inverse. Currently using left inverse.
        return self.linverse(*args, **kwargs)

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

    # Vector Voting

    def gaussian_blur(self, sigma=1, kernel_size=5):
        """Gausssian blur the displacement field to reduce any unsmoothness
        """
        pad = (kernel_size - 1) // 2
        if kernel_size % 2 == 0:
            pad = (pad, pad+1, pad, pad+1)
        else:
            pad = (pad,)*4
        smoothing = GaussianSmoothing(channels=2, kernel_size=kernel_size,
                                      sigma=sigma, device=self.device)
        return smoothing(F.pad(self, pad, mode='reflect'))

    def vote(self, softmin_temp=1, blur_sigma=1):
        """Produce a single, consensus displacement field from a set of
        displacement fields

        The resulting displacement field represents displacements that are
        closest to the most constent majority of the fields.
        This effectively allows the fields to differentiably vote on the
        displacement that is most likely to be correct.

        Args:
            self: DisplacementField of shape (N, 2, H, W)
            softmin_temp (float): temperature of softmin to use
            blur_sigma (float): std dev of the Gaussian kernel by which to blur
                the inputs; None means no blurring

        Returns:
            DisplacementField of shape (1, 2, H, W) containing the vector
            voting result
        """
        if self.ndimension() != 4:
            raise ValueError('Vector voting only implemented on displacement '
                             'fields with 4 dimensions. The input has {}.'
                             .format(self.ndimension()))
        n, two, *shape = self.shape
        if n % 2 == 0:
            raise ValueError('Cannot vetor vote on an even number of '
                             'displacement fields: {}'.format(n))
        elif n == 1:
            return self
        m = (n + 1) // 2  # smallest number that constututes a majority
        blurred = self.gaussian_blur(sigma=blur_sigma) if blur_sigma else self

        # compute distances for all pairs of fields
        dists = torch.zeros((n, n, *shape)).to(device=blurred.device)
        for i in range(n):
            for j in range(i):
                dists[i, j] = dists[j, i] \
                    = blurred[i].distance(blurred[j])

        # compute mean distance for all m-tuples
        mtuples = list(combinations(range(n), m))
        mtuple_avg = []
        for mtuple in mtuples:
            delta = torch.stack([
                dists[i, j] for i, j in combinations(mtuple, 2)
            ]).mean(dim=0)
            mtuple_avg.append(delta)
        mavg = torch.stack(mtuple_avg)

        # compute weights for mtuples: smaller mean distance -> higher weight
        mt_weights = (-mavg/softmin_temp).softmax(dim=0)

        # assign mtuple weights back to individual fields
        field_weights = torch.zeros((n, *shape)).to(device=mt_weights.device)
        for i, mtuple in enumerate(mtuples):
            for j in mtuple:
                field_weights[j] += mt_weights[i]
        field_weights = field_weights / field_weights.sum(dim=0, keepdim=True)

        return (self * field_weights.unsqueeze(-3)).sum(dim=0, keepdim=True)


torch.Field = DisplacementField

######################
# possible TODO list #
######################

"""BUG: 2 ** disp_field --> RecursionError
"""

"""State variables
- Direction convention: Euler/Lagrange (aka Push/Pull) conventions
- Scale convention: [-1,1], [0, 1], [0, size], [-0.5, size-0.5]
- Identity convention: Mapping vs. Displacement (with / without identity)
- Dimension order: (B, C, [Z], Y, X), (B, C, X, Y, [Z]),
                   (B, [Z], Y, X, C), (B, X, Y, Z, C)
                   B = Batch numbers, C = Component
- Component order: XYZ or ZYX

Functions for conversion between states
"""

"""Unit testing
"""
