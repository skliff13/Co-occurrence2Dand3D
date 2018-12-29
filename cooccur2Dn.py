# -*- coding: utf-8 -*-
import numpy as np
from scipy.signal import correlate2d


def cooccur2Dn(gray_image2d, i_range=(0, 1.), i_bins=8, dists=(1,), num_dots=2, mask=None, econ=False):
    """Calculates extended IID and IIID co-occurrence matrix of a 2D gray-level image.

    Extended IIID co-occurrence matrices count occurrences of triplets of pixels in the target
    image which have certain properties. The matrix considers both pixels' intensity (I) and distance between
    the pixels of each triplet (D).

    See paper for details.
        Kovalev V., Dmitruk A., Safonau I., Frydman M., Shelkovich S. (2011) A Method for Identification and
        Visualization of Histological Image Structures Relevant to the Cancer Patient Conditions.
        In: Real P., Diaz-Pernil D., Molina-Abril H., Berciano A., Kropatsch W. (eds) Computer Analysis of
        Images and Patterns. CAIP 2011. Lecture Notes in Computer Science, vol 6854. Springer, Berlin, Heidelberg

    Args:
        gray_image2d (ndarray): 2D numpy array representing the image.
        i_range (tuple): Min and max values indicating the intensity (I) range. Defaults to (0, 1.)
        i_bins (int): Number of intensity (I) bins. Defaults to 8.
        dists (tuple): Distances between pixels to consider. Defaults to (1,).
        num_dots (int): Can be either 2 for pairs or 3 for pixels triplets.
        mask (ndarray): Region-of-interest (ROI) mask. Only pixels which correspond to positive 'mask' elements
            are considered.
        econ (bool): When set to True, only 0, 45, 90 and 135-degree connections between the first two pixels
            are considered.
            Makes algorithm run faster when large distances between pixels are considered. Defaults to False.

    Return:
        ndarray: Co-occurrence matrix. Dimensionality depends on the binning and number of distances considered.

    """

    gray_image2d = gray_image2d.astype(float)

    if mask is not None:
        gray_image2d, mask = __crop_using_mask(gray_image2d, mask, dists)

    binned_i = __to_bins(gray_image2d, i_range, i_bins)
    if mask is not None:
        binned_i[mask == 0] = -1

    offsets = calc_offsets(dists, econ)

    all_bins = [i_bins] * num_dots
    bins_prod = int(np.prod(all_bins))
    result = np.zeros((len(dists) * bins_prod))

    for i in range(offsets.shape[0]):
        sub_result = __process_offset(i_bins, binned_i, bins_prod, dists, i, offsets, num_dots)
        result += sub_result

    all_dims = all_bins.copy()
    all_dims.append(len(dists))
    dims = []
    for dim in all_dims:
        if dim > 1:
            dims.append(dim)
    dims.reverse()

    result = result.reshape(tuple(dims))

    return result


def __process_offset(i_bins, binned_i, bins_prod, dists, i, offsets, num_dots):
    dist_idx = dists.index(offsets[i, 0])
    dist_offs = offsets[offsets[:, 0] == dists[dist_idx], :].copy()
    dist_offs = np.append(dist_offs, [[0] * offsets.shape[1]], axis=0)

    to_crop = (max(-dist_offs[:, 2::2].flatten()), max(dist_offs[:, 2::2].flatten()),
               max(-dist_offs[:, 1::2].flatten()), max(dist_offs[:, 1::2].flatten()))

    ii = (to_crop[0], binned_i.shape[0] - to_crop[1])
    jj = (to_crop[2], binned_i.shape[1] - to_crop[3])
    b0 = __crop_roi(binned_i, ii, jj)

    bs = []
    for ndot in range(2, num_dots + 1):
        d = (ndot - 2) * 2
        ii1 = (to_crop[0] + offsets[i, 2 + d], binned_i.shape[0] - to_crop[1] + offsets[i, 2 + d])
        jj1 = (to_crop[2] + offsets[i, 1 + d], binned_i.shape[1] - to_crop[3] + offsets[i, 1 + d])
        b1 = __crop_roi(binned_i, ii1, jj1)
        bs.append(b1)

    comatrix_bins = __map_matrix_bins(i_bins, b0, bs, num_dots)

    hist, _ = np.histogram(comatrix_bins, bins=range(1, bins_prod + 2, 1))
    hist_shifted = np.zeros((len(dists) * bins_prod))
    hist_shifted[dist_idx * bins_prod:(dist_idx + 1) * bins_prod] = hist
    return hist_shifted


def __map_matrix_bins(i_bins, b0, bs, num_dots):
    feature_bins = np.zeros((b0.shape[0], b0.shape[1], num_dots)).astype(np.int8)
    feature_bins[:, :, 0] = b0
    for ndot in range(2, num_dots + 1):
        feature_bins[:, :, ndot - 1] = bs[ndot - 2]

    feature_bins = np.sort(feature_bins, axis=2)
    min_values = np.min(feature_bins, axis=2)

    comatrix_bins = np.ones((feature_bins.shape[0], feature_bins.shape[1]))
    for ndot in range(1, num_dots + 1):
        comatrix_bins += (feature_bins[:, :, ndot - 1].astype(float) - 1) * i_bins**(ndot - 1)

    comatrix_bins[min_values < 0] = -1
    comatrix_bins = comatrix_bins.flatten()
    comatrix_bins = comatrix_bins[comatrix_bins > 0]

    return comatrix_bins


def __crop_roi(b, ii, jj):
    b0 = b[ii[0]:ii[1], jj[0]:jj[1]]
    return b0


def __crop_using_mask(im, mask, dists):
    mask[mask < 0] = 0

    d = 1 + max(dists)
    bounds = []
    proj_dims = [1, 0]
    for dim in range(len(im.shape)):
        projection = np.sum(mask, axis=proj_dims[dim]).flatten()
        idx = np.argwhere(projection > 0)
        first = int(max(0, idx[0] - d))
        last = int(min(len(projection), idx[-1] + d))
        bounds.append((first, last))

    im = im[bounds[0][0]:bounds[0][1], bounds[1][0]:bounds[1][1]]
    mask = mask[bounds[0][0]:bounds[0][1], bounds[1][0]:bounds[1][1]]

    return im, mask


def __to_bins(feature, feature_range, feature_bins):
    binned = (feature - feature_range[0]) / float(feature_range[1] - feature_range[0])
    binned = (np.floor(binned * feature_bins) + 1).astype(np.int8)
    binned[binned < 1] = 1
    binned[binned > feature_bins] = feature_bins
    return binned


def __calculate_gradients(gray_image2d):
    sobel_operator = np.asarray([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    grad_x = correlate2d(gray_image2d, sobel_operator, mode='same', boundary='symm')
    grad_y = correlate2d(gray_image2d, np.transpose(sobel_operator), mode='same', boundary='symm')
    return grad_x, grad_y


def calc_offsets(dists, econ):
    if not econ:
        return calc_offsets_all(dists)
    else:
        return calc_offsets_econ(dists)


def calc_offsets_econ(dists):
    angles = np.pi / 4 * np.asarray([0, 1, 2, 3])
    s3 = 3. ** 0.5

    offsets = []
    for d in dists:
        for angle in angles:
            x = round(d * np.cos(angle))
            y = round(d * np.sin(angle))
            x1 = round(x / 2. - s3 / 2. * y)
            y1 = round(y / 2. + s3 / 2. * x)
            offsets.append([d, x, y, x1, y1])

    offsets.sort()
    offsets = np.asarray(offsets).astype(int)
    return offsets


def calc_offsets_all(dists):
    max_dist = max(dists)
    offsets = []

    y = 0
    for x in range(1, max_dist + 1):
        __add_offset(offsets, dists, x, y)

    for x in range(-max_dist, max_dist + 1):
        for y in range(1, max_dist + 1):
            __add_offset(offsets, dists, x, y)

    offsets.sort()
    offsets = np.asarray(offsets).astype(int)
    return offsets


def __add_offset(offsets, dists, x, y):
    s3 = 3. ** 0.5
    d = round((x ** 2 + y ** 2) ** 0.5)
    if d in dists:
        x1 = round(x / 2. - s3 / 2. * y)
        y1 = round(y / 2. + s3 / 2. * x)
        offsets.append([d, x, y, x1, y1])
