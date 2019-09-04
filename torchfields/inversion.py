import torch
import torch.nn.functional as F

from .utils import ensure_dimensions


# Inversion helper functions

def _tensor_min(*args):
    """Elementwise minimum of a sequence of tensors"""
    minimum, *rest = args
    for arg in rest:
        minimum = minimum.min(arg)
    return minimum


def _tensor_max(*args):
    """Elementwise maximum of a sequence of tensors"""
    maximum, *rest = args
    for arg in rest:
        maximum = maximum.max(arg)
    return maximum


class _BackContig(torch.autograd.Function):
    """Ensure that the gradient is contiguous in the backward pass"""
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.contiguous()


_back_contig = _BackContig.apply


def _pad(inp, padding=None):
    """Pads the field just enough to eliminate border effects"""
    if padding is None:
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
            # ensure that the new field is square (that is, newH = newW)
            newH, newW = pad_yl + H + pad_yh, pad_xl + W + pad_xh
            if newH > newW:
                pad_xh += newH - newW
            elif newW > newH:
                pad_yh += newW - newH
            padding = (pad_xl, pad_xh, pad_yl, pad_yh)
    return (F.pad(inp.pixels(), padding, mode='replicate').field()
            .from_pixels(), padding)


def _unpad(inp, padding):
    """Crops the field back to its original size"""
    p_xl, p_xh, p_yl, p_yh = padding
    p_xh = inp.shape[-1] - p_xh
    p_yh = inp.shape[-2] - p_yh
    return inp.pixels()[..., p_yl:p_yh, p_xl:p_xh].from_pixels()


def _fold(inp):
    """Collapse the matrix at each pixel onto the local neighborhood

    The input to this function is a spatial tensor in which the
    entry contained in every spatial pixel is itself a small matrix.
    The values of this matrix correspond to the neighborhood of that
    pixel. For instance, the value at the center of the matrix
    corresponds to the pixel itself, whereas the value above and to
    the left of the center corresponds to the pixel's upper left
    neighbor, and so on.

    This function collapses this into a spatial tensor with scalar
    values by summing the respective values corresponding to each
    pixel.
    """
    pad = (0, (inp.shape[2] + 1) % 2,  # last dimension
           0, (inp.shape[1] + 1) % 2)  # second to last dimension
    res = F.pad(_back_contig(inp), pad)
    res = F.fold(
        res.view(1, res.shape[0]*res.shape[1]*res.shape[2], -1).contiguous(),
        output_size=inp.shape[3:], kernel_size=inp.shape[1:3],
        padding=((inp.shape[1])//2, (inp.shape[2])//2))
    return res


@torch.no_grad()
def _winding_number(Px, Py, v00, v01, v10, v11, eps3):
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


#############################
# Displacement Field Inverses
#############################

@ensure_dimensions(ndimensions=4, arg_indices=(0), reverse=True)
def linverse(self, autopad=True):
    r"""Return a left inverse approximation for the displacement field

    Given a displacement field `f`, its left inverse is a displacement
    field `g` such that
    `g(f) ~= identity`

    In other words
    :math:`f_{inv} = \argmin_{g} |g(f)|^2`
    """
    if len(self.shape) != 4 or self.shape[0] > 1:
        raise NotImplementedError('Left inverse is currently implemented '
                                  'only for single-batch fields. '
                                  'Received batch size {}'
                                  .format(','.join(
                                      str(n) for n in self.shape[:-3])))
    # comparison to 0
    eps1 = 2**(-51) if self.dtype is torch.double else 2**(-23)
    # denominator fudge factor to avoid dividing by 0
    eps2 = eps1 * 2**(-10)
    # tolarance for point containment in a quadrilateral
    eps3 = 2**-16

    try:
        # pad the field
        if autopad:
            field, padding = _pad(self)
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
            v_min = _tensor_min(v00, v01, v10, v11).floor()
            v_min.y.clamp_(0, field.shape[-2] - 1)
            v_min.x.clamp_(0, field.shape[-1] - 1)
            v_max = _tensor_max(v00, v01, v10, v11).floor() + 1
            v_max.y.clamp_(0, field.shape[-2] - 1)
            v_max.x.clamp_(0, field.shape[-1] - 1)
            # d_x and d_y are the largest spans in x and y
            d = (v_max - v_min).max_vector().max(0)[0].long()
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
                  / (2*a).abs().clamp(min=eps2))
        # corner case when a == 0 (reduces to `b*j + c = 0`)
        j_temp = j_temp.where(a.abs() > eps1, c.abs()/b.abs().clamp(min=eps2))
        a = b = c = None  # del a, b, c
        # get i from j_temp
        i = ((uy - v00.y + (v00.y - v01.y) * j_temp).abs()
             / (-v00.y + v10.y + (v00.y - v01.y - v10.y + v11.y) * j_temp)
             .abs().clamp(min=eps2))
        j_temp = None  # del j_temp
        # j has significantly smaller rounding error for near-trapezoids
        j = ((ux - v00.x + (v00.x - v10.x) * i).abs()
             / (-v00.x + v01.x + (v00.x - v10.x - v01.x + v11.x) * i)
             .abs().clamp(min=eps2))
        # winding_number > 0 means point is contained in the quadrilateral
        wn = _winding_number(ux, uy, v00, v01, v10, v11, eps3)
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
                   .view(2, -1).flip(0).round().long().contiguous())
        v_min = d = None
        inv = inv.view(3, d_y, d_x, -1).permute(3, 0, 1, 2).contiguous()
        # construct sparse tensor and use `to_dense` to arrange vectors
        SparseTensor = (torch.cuda.sparse.FloatTensor if self.is_cuda
                        else torch.sparse.FloatTensor)
        inv = SparseTensor(indices, inv, (*field.shape[-2:], 3, d_y, d_x),
                           device=inv.device)
        inv = inv.to_dense().permute(2, 3, 4, 0, 1)
        # fold the d_y by d_x neighborhoods by summing the overlaps
        inv = _fold(inv)
        # divide each pixel by number of contributions to get an average
        inv = inv[:, :2] / inv[:, 2:].clamp(min=1.)

        # crop back to original shape
        if autopad:
            inv = _unpad(inv.field(), padding)
        return inv
    except RuntimeError:
        # In case this is an out-of-memory error, clear temporary tensors
        self = field = mapping = v_min = v_max = d = d_x = d_y = None
        u = ux = uy = v00 = v01 = v10 = v11 = wn = None
        a = b = c = j_temp = i = j = None
        mask = inv = indices = None
        raise


def rinverse(self, *args, **kwargs):
    r"""Return a right inverse approximation for the displacement field

    Given a displacement field `f`, its right inverse is a displacement
    field `g` such that
    `f(g) ~= identity`

    In other words
    :math:`f_{inv} = \argmin_{g} |f(g)|^2`
    """
    raise NotImplementedError
