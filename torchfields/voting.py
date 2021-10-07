import torch
import torch.nn.functional as F


############################
# Vector vote implemtation
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
        pad = (pad, pad + 1, pad, pad + 1)
    else:
        pad = (pad,) * 4
    padded = F.pad(self, pad, mode="reflect")
    mu = (kernel_size - 1) / 2
    x = torch.stack(torch.meshgrid([torch.arange(kernel_size).to(self)] * 2))
    kernel = torch.exp((-(((x - mu) / sigma) ** 2)) / 2)
    kernel = kernel.prod(dim=0) / (2 * math.pi * sigma ** 2)
    kernel = kernel / kernel.sum()  # renormalize to get unit product
    kernel = kernel.expand(2, 1, *kernel.shape)
    return F.conv2d(padded, weight=kernel, groups=2)


def get_vote_shape(self):
    """Consistently split shape into N fields & shape of field
    """
    n, _, *shape = self.shape
    return n, shape


def get_subset_size(self, subset_size=None):
    """Compute smallest majority of self is subset_size not set
    """
    n, _ = self.get_vote_shape()
    m = (n + 1) // 2  # smallest number that constututes a majority
    if subset_size is not None:
        m = subset_size
    return m


def get_vote_subsets(self, subset_size=None):
    """Compute list of majority subsets to use in vote
    """
    n, _ = self.get_vote_shape()
    m = self.get_subset_size(subset_size=subset_size)
    from itertools import combinations

    subset_tuples = list(combinations(range(n), m))
    return subset_tuples


def get_vote_weights(self, softmin_temp=1, blur_sigma=1, subset_size=None):
    """Calculate per field weights for batch of displacement fields, indicating 
    which fields should be considered consensus.

    Args:
        self: DisplacementField of shape (N, 2, H, W)
        softmin_temp (float): temperature of softmin to use
        blur_sigma (float): std dev of the Gaussian kernel by which to blur
            the softmin inputs. Note that the outputs are not blurred.
            None or 0 means no blurring.
        subset_size (int): number of members to each set for comparison

    Returns:
        per field weight (torch.Tensor): (N, 1, H, W)
    """
    from itertools import combinations

    if self.ndimension() != 4:
        raise ValueError(
            "Vector vote is only implemented on "
            "displacement fields with 4 dimensions. "
            "The input has {}.".format(self.ndimension())
        )
    n, shape = self.get_vote_shape()
    subset_size = self.get_subset_size(subset_size=subset_size)
    if n == 1:
        return torch.ones((1, *shape)).to(self)
    if n == subset_size:
        return torch.ones((1, *shape)).to(self) / n
    # elif n % 2 == 0:
    #     raise ValueError('Cannot vetor vote on an even number of '
    #                      'displacement fields: {}'.format(n))
    blurred = self.gaussian_blur(sigma=blur_sigma) if blur_sigma else self
    mtuples = self.get_vote_subsets(subset_size=subset_size)

    # compute distances for all pairs of fields
    dists = torch.zeros((n, n, *shape)).to(device=blurred.device)
    for i in range(n):
        for j in range(i):
            dists[i, j] = dists[j, i] = blurred[i].distance(blurred[j])

    # compute mean distance for all majority tuples
    mtuple_avg = []
    for mtuple in mtuples:
        delta = torch.stack([dists[i, j] for i, j in combinations(mtuple, 2)]).mean(
            dim=0
        )
        mtuple_avg.append(delta)
    mavg = torch.stack(mtuple_avg)

    # compute weights for mtuples: smaller mean distance -> higher weight
    # computing softmin explicitly for access to numerator & denominator
    # equivalent to:
    # (-mavg/softmin_temp).softmax(dim=0) == mt_weights_num / mt_weights_den
    mt_weights_num = torch.exp(-mavg / softmin_temp)
    mt_weights_den = mt_weights_num.sum(dim=0, keepdim=True)

    mt_weights = mt_weights_num / mt_weights_den
    # assign mtuple weights back to individual fields
    field_weights = torch.zeros((n, *shape)).to(device=mt_weights.device)
    for i, mtuple in enumerate(mtuples):
        for j in mtuple:
            field_weights[j] += mt_weights[i]
    # rather than use m, prefer sum for sum precision
    elements_per_subset = field_weights.sum(dim=0, keepdim=True)
    field_weights = field_weights / elements_per_subset
    return field_weights


def vote(self, softmin_temp=1, blur_sigma=1, subset_size=None):
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
        subset_size (int): number of members to each subset

    Returns:
        DisplacementField of shape (1, 2, H, W) containing the vector
        vote result
    """
    field_weights = self.get_vote_weights(
        softmin_temp=softmin_temp, blur_sigma=blur_sigma, subset_size=subset_size
    )
    return (self * field_weights.unsqueeze(-3)).sum(dim=0, keepdim=True)


def get_vote_weights_with_distances(
    self, distances, softmin_temp=1, blur_sigma=1, subset_size=None
):
    """Calculate consensus field from batch of displacement fields along with distances.
    Voting proceeds as normal, until it comes time to distribute the weight of each
    subset amongst its constitute vectors. The distribution is now based on the 
    distances of each vector, with distances further away making a vector contribute
    less to consensus than nearer distance vectors in the subset.

    Weights should be proportional to get_vote_weights if distances are identical.

    Args:
        self: DisplacementField of shape (N, 2, H, W)
        distances: Tensor of shape (N, H, W)
        softmin_temp (float): temperature of softmin to use
        blur_sigma (float): std dev of the Gaussian kernel by which to blur
            the softmin inputs. Note that the outputs are not blurred.
            None or 0 means no blurring.
        subset_size (int): number of members to each set for comparison

    Returns:
        per field weight (torch.Tensor): (N, 1, H, W)
    """
    from itertools import combinations

    if self.ndimension() != 4:
        raise ValueError(
            "Vector vote is only implemented on "
            "displacement fields with 4 dimensions. "
            "The input has {}.".format(self.ndimension())
        )
    n, shape = self.get_vote_shape()
    subset_size = self.get_subset_size(subset_size=subset_size)
    if n == 1:
        return torch.ones((1, *shape)).to(self)
    blurred = self.gaussian_blur(sigma=blur_sigma) if blur_sigma else self
    # distances = distances.gaussian_blur(sigma=blur_sigma) if blur_sigma else distances
    subset_tuples = self.get_vote_subsets(subset_size=subset_size)

    # compute mean of mixture distribution for all subset tuples
    subset_avg = {}
    for subset in subset_tuples:
        s_avg = torch.stack([blurred[i] for i in subset]).mean(dim=0)
        subset_avg[subset] = s_avg

    # compute standard deviations of mixture distribution for all subset tuples with var=0
    subset_std = []
    for subset in subset_tuples:
        s_moment_sum = torch.stack([blurred[i].pow(2) for i in subset])
        s_var = (s_moment_sum - subset_avg[subset].pow(2)).mean(dim=0)
        subset_std.append(s_var.abs().sqrt())
    subset_std_dist = torch.stack(subset_std).pow(2).sum(dim=-3).sqrt()

    # compute weights for subset_tuples: smaller variance -> higher weight
    # computing softmin explicitly for access to numerator & denominator
    # equivalent to:
    # subset_weights == (-subset_std_dist/softmin_temp).softmax(dim=0)
    subset_weights_num = torch.exp(-subset_std_dist / softmin_temp)
    subset_weights_den = subset_weights_num.sum(dim=0, keepdim=True)

    subset_weights = subset_weights_num / subset_weights_den
    # assign subset weights back to individual fields
    # use distances to partition the weights: larger distance -> less weight
    field_weights = torch.zeros((n, *shape)).to(device=subset_weights.device)
    for i, subset in enumerate(subset_tuples):
        dists = distances[subset, ...]
        weights = (1.0 / dists) * (1.0 / (1.0 / dists).sum(dim=0))
        for k, j in enumerate(subset):
            field_weights[j] += subset_weights[i] * weights[k]
    return field_weights


def vote_with_distances(
    self, distances, softmin_temp=1, blur_sigma=1, subset_size=None
):
    """Produce a single, consensus displacement field from a batch of
    displacement fields along with a distance measure that weights further
    fields less in the consensus.

    Args:
        self: DisplacementField of shape (N, 2, H, W)
        distances: Tensor of shape (N, 1, H, W)
        softmin_temp (float): temperature of softmin to use
        blur_sigma (float): std dev of the Gaussian kernel by which to blur
            the softmin inputs. Note that the outputs are not blurred.
            None or 0 means no blurring.
        subset_size (int): number of members to each subset

    Returns:
        DisplacementField of shape (1, 2, H, W) containing the vector
        vote result
    """
    field_weights = self.get_vote_weights_with_distances(
        softmin_temp=softmin_temp,
        distances=distances,
        blur_sigma=blur_sigma,
        subset_size=subset_size,
    )
    return (self * field_weights.unsqueeze(-3)).sum(dim=0, keepdim=True)


def get_vote_weights_with_variances(
    self, var, softmin_temp=1, blur_sigma=1, subset_size=None
):
    """Calculate consensus field from batch of displacement fields along with variances.
    Each vector within self is treated as the mean of a distribution with isotropic 
    variance for the corresponding location in variances. A subset of vectors is
    considered a mixture distribution. We assign higher weight to mixture distributions 
    with lower variances.

    Weights should be proportional to get_vote_weights if variances are zero.

    Args:
        self: DisplacementField of shape (N, 2, H, W)
        var: Tensor of shape (N, 1, H, W)
        softmin_temp (float): temperature of softmin to use
        blur_sigma (float): std dev of the Gaussian kernel by which to blur
            the softmin inputs. Note that the outputs are not blurred.
            None or 0 means no blurring.
        subset_size (int): number of members to each set for comparison

    Returns:
        per field weight (torch.Tensor): (N, 1, H, W)
    """
    from itertools import combinations

    if self.ndimension() != 4:
        raise ValueError(
            "Vector vote is only implemented on "
            "displacement fields with 4 dimensions. "
            "The input has {}.".format(self.ndimension())
        )
    n, shape = self.get_vote_shape()
    subset_size = self.get_subset_size(subset_size=subset_size)
    if n == 1:
        return torch.ones((1, *shape)).to(self)
    if n == subset_size:
        return torch.ones((1, *shape)).to(self) / n
    blurred = self.gaussian_blur(sigma=blur_sigma) if blur_sigma else self
    variances = torch.cat([var, var], dim=1).field()
    variances = variances.gaussian_blur(sigma=blur_sigma) if blur_sigma else variances
    subset_tuples = self.get_vote_subsets(subset_size=subset_size)

    # compute mean of mixture distribution for all subset tuples
    subset_avg = {}
    for subset in subset_tuples:
        s_avg = torch.stack([blurred[i] for i in subset]).mean(dim=0)
        subset_avg[subset] = s_avg

    # compute standard deviations of mixture distribution for all subset tuples
    subset_std = []
    for subset in subset_tuples:
        s_moment_sum = torch.stack([variances[i] + blurred[i].pow(2) for i in subset])
        s_var = (s_moment_sum - subset_avg[subset].pow(2)).mean(dim=0)
        subset_std.append(s_var.abs().sqrt())
    subset_std_dist = torch.stack(subset_std).pow(2).sum(dim=-3).sqrt()

    # compute weights for subset_tuples: smaller variance -> higher weight
    # computing softmin explicitly for access to numerator & denominator
    # equivalent to:
    # subset_weights == (-subset_std_dist/softmin_temp).softmax(dim=0)
    subset_weights_num = torch.exp(-subset_std_dist / softmin_temp)
    subset_weights_den = subset_weights_num.sum(dim=0, keepdim=True)

    subset_weights = subset_weights_num / subset_weights_den
    # assign subset weights back to individual fields
    field_weights = torch.zeros((n, *shape)).to(device=subset_weights.device)
    for i, subset in enumerate(subset_tuples):
        for j in subset:
            field_weights[j] += subset_weights[i]
    # rather than use subset_size, prefer sum for sum precision
    elements_per_subset = field_weights.sum(dim=0, keepdim=True)
    field_weights = field_weights / elements_per_subset
    return field_weights


def vote_with_variances(self, var, softmin_temp=1, blur_sigma=1, subset_size=None):
    """Produce a single, consensus displacement field from a batch of
    distributions, with displacement fields as mean and variances.

    Args:
        self: DisplacementField of shape (N, 2, H, W)
        var: Tensor of shape (N, 1, H, W)
        softmin_temp (float): temperature of softmin to use
        blur_sigma (float): std dev of the Gaussian kernel by which to blur
            the softmin inputs. Note that the outputs are not blurred.
            None or 0 means no blurring.
        subset_size (int): number of members to each subset

    Returns:
        DisplacementField of shape (1, 2, H, W) containing the vector
        vote result
    """
    field_weights = self.get_vote_weights_with_variances(
        softmin_temp=softmin_temp,
        var=var,
        blur_sigma=blur_sigma,
        subset_size=subset_size,
    )
    return (self * field_weights.unsqueeze(-3)).sum(dim=0, keepdim=True)
