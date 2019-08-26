from functools import wraps


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


###########################################################################
# Decorators to convert the inputs and outputs of DisplacementField methods
###########################################################################

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
