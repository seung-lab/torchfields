import torch
import torch.nn.functional as F


############################
# Vector voting implemtation
############################

def gaussian_blur(self, sigma=1, kernel_size=5):
    """Gausssian blur the displacement field to reduce any unsmoothness
    Adapted from https://bit.ly/2JO7CCP
    """
    import math
    if sigma == 0:
        return self.clone()
    pad = (kernel_size - 1) // 2
    if kernel_size % 2 == 0:
        pad = (pad, pad+1, pad, pad+1)
    else:
        pad = (pad,)*4
    padded = F.pad(self, pad, mode='reflect')
    mu = (kernel_size - 1) / 2
    x = torch.stack(torch.meshgrid([torch.arange(kernel_size).to(self)]*2))
    kernel = torch.exp((-((x - mu) / sigma) ** 2) / 2)
    kernel = kernel.prod(dim=0) / (2 * math.pi * sigma**2)
    kernel = kernel / kernel.sum()  # renormalize to get unit product
    kernel = kernel.expand(2, 1, *kernel.shape)
    return F.conv2d(padded, weight=kernel, groups=2)


def vote(self, softmin_temp=1, blur_sigma=1):
    """Produce a single, consensus displacement field from a batch of
    displacement fields

    The resulting displacement field represents displacements that are
    closest to the most consistent majority of the fields.
    This effectively allows the fields to differentiably vote on the
    displacement that is most likely to be correct.

    Args:
        self: DisplacementField of shape (N, 2, H, W)
        softmin_temp (float): temperature of softmin to use
        blur_sigma (float): std dev of the Gaussian kernel by which to blur
            the softmin inputs. Note that the outputs are not blurred.
            None or 0 means no blurring.

    Returns:
        DisplacementField of shape (1, 2, H, W) containing the vector
        voting result
    """
    field_weights = get_field_weights(self, 
                                      softmin_temp=softmin_temp,
                                      blur_sigma=blur_sigma)
    partition = field_weights.sum(dim=0, keepdim=True)
    field_weights = field_weights / partition
    return (self * field_weights.unsqueeze(-3)).sum(dim=0, keepdim=True)

def get_field_weights(field, softmin_temp=1, blur_sigma=1):
    """Calculate softmin weights for set of fields, indicating which fields
    should be considered consensus.

    Args:
        field: DisplacementField of shape (N, 2, H, W)
        softmin_temp (float): temperature of softmin to use
        blur_sigma (float): std dev of the Gaussian kernel by which to blur
            the softmin inputs. Note that the outputs are not blurred.
            None or 0 means no blurring.

    Returns:
        torch.Tensor of shape (N, 1, H, W) containing the field weights 
    """
    from itertools import combinations
    if field.ndimension() != 4:
        raise ValueError('Vector voting is only implemented on '
                         'displacement fields with 4 dimensions. '
                         'The input has {}.'.format(field.ndimension()))
    n, _two_, *shape = field.shape
    if n == 1:
        return field
    elif n % 2 == 0:
        raise ValueError('Cannot vetor vote on an even number of '
                         'displacement fields: {}'.format(n))
    m = (n + 1) // 2  # smallest number that constututes a majority
    blurred = field.gaussian_blur(sigma=blur_sigma) if blur_sigma else field

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
    return field_weights
