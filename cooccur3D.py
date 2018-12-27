# -*- coding: utf-8 -*-
import numpy as np
from skimage import io
from scipy.signal import correlate


def cooccur3D(gray_image3d, i_range=(0, 1), i_bins=8, g_range=(0, 1), g_bins=1, a_bins=1, dists=(1,), mask=None,
              econ=False, z2xy=1.0):
    """Calculates extended multi-sort IIGGAD co-occurrence matrix of a 3D gray-level image.

        Extended multi-sort IIGGAD co-occurrence matrices count occurrences of pairs of pixels in the target
        image which have certain properties. The matrix considers both pixels' intensity (I), gradient magnitudes (G),
        angle between gradient directions (A) and distance between the two pixels (D).

    See paper for details.
        V. A. Kovalev, F. Kruggel, H.-J. Gertz and D. Y. von Cramon, "Three-dimensional texture analysis of
        MRI brain datasets," in IEEE Transactions on Medical Imaging, vol. 20, no. 5, pp. 424-433, May 2001.
        doi: 10.1109/42.925295

    Args:
        gray_image3d (ndarray): 3D numpy array representing the image.
        i_range (tuple): Min and max values indicating the intensity (I) range. Defaults to (0, 1)
        i_bins (int): Number of intensity (I) bins. Defaults to 8.
        g_range (tuple): Min and max values indicating the gradient magnitude (G) range. Defaults to (0, 1)
        g_bins (int): Number of gradient magnitude (G) bins. Defaults to 1.
        a_bins (int): Number of angle (A) bins. Defaults to 1.
        dists (tuple): Distances between pixels to consider. Defaults to (1,).
        mask (ndarray): Region-of-interest (ROI) mask. Only pixels which correspond to positive 'mask' elements
            are considered.
        econ (bool): When set to True, only connections between pixels under degrees multiple of 45 are considered.
            Makes algorithm run faster when large distances between pixels are considered. Defaults to False.
        z2xy (float): Ratio of voxel size along Z axis to its size along XY plane. Defaults to 1.0.

    Return:
        ndarray: Multi-sort co-occurrence matrix.
            Dimensionality depends on the binning and number of distances considered.

    """

    gray_image3d = gray_image3d.astype(float)

    if mask is not None:
        gray_image3d, mask = __crop_using_mask(gray_image3d, mask, dists, z2xy)

    binned_i = __to_bins(gray_image3d, i_range, i_bins)
    if mask is not None:
        binned_i[mask == 0] = -1

    grad_x, grad_y, grad_z = __calculate_gradients(gray_image3d, z2xy)
    grad_magnitude = (grad_x**2 + grad_y**2 + grad_z**2)**0.5

    binned_g = __to_bins(grad_magnitude, g_range, g_bins)

    offsets = calc_offsets(dists, econ, z2xy)

    all_bins = [i_bins, i_bins, g_bins, g_bins, a_bins]
    bins_prod = int(np.prod(all_bins))
    result = np.zeros((len(dists) * bins_prod))

    for i in range(offsets.shape[0]):
        sub_result = __process_offset(a_bins, all_bins, binned_g, binned_i, bins_prod, dists, grad_x, grad_y, i, offsets)
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


def __process_offset(a_bins, all_bins, binned_g, binned_i, bins_prod, dists, grad_x, grad_y, i, offsets):
    dist_idx = dists.index(offsets[i, 0])
    dist_offs = offsets[offsets[:, 0] == dists[dist_idx], :].copy()
    dist_offs = np.append(dist_offs, [[0] * offsets.shape[1]], axis=0)

    to_crop = (max(-dist_offs[:, 2]), max(dist_offs[:, 2]), max(-dist_offs[:, 1]), max(dist_offs[:, 1]),
               max(-dist_offs[:, 3]), max(dist_offs[:, 3]))

    ii = (to_crop[0], binned_i.shape[0] - to_crop[1])
    jj = (to_crop[2], binned_i.shape[1] - to_crop[3])
    kk = (to_crop[4], binned_i.shape[2] - to_crop[5])
    b0, g0, x0, y0 = __crop_roi(binned_i, binned_g, grad_x, grad_y, ii, jj, kk)

    ii1 = (to_crop[0] + offsets[i, 2], binned_i.shape[0] - to_crop[1] + offsets[i, 2])
    jj1 = (to_crop[2] + offsets[i, 1], binned_i.shape[1] - to_crop[3] + offsets[i, 1])
    kk1 = (to_crop[4] + offsets[i, 3], binned_i.shape[2] - to_crop[5] + offsets[i, 3])
    b1, g1, x1, y1 = __crop_roi(binned_i, binned_g, grad_x, grad_y, ii1, jj1, kk1)
    ba = __calc_angular_bins(a_bins, x0, x1, y0, y1)

    comatrix_bins = __map_matrix_bins(all_bins, b0, b1, ba, g0, g1)

    hist, _ = np.histogram(comatrix_bins, bins=range(1, bins_prod + 2, 1))
    hist_shifted = np.zeros((len(dists) * bins_prod))
    hist_shifted[dist_idx * bins_prod:(dist_idx + 1) * bins_prod] = hist
    return hist_shifted


def __map_matrix_bins(all_bins, b0, b1, ba, g0, g1):
    feature_bins = np.zeros((b0.shape[0], b0.shape[1], b0.shape[2], 5)).astype(np.int8)
    feature_bins[:, :, :, 0] = b0
    feature_bins[:, :, :, 1] = b1
    feature_bins[:, :, :, 2] = g0
    feature_bins[:, :, :, 3] = g1
    feature_bins[:, :, :, 4] = ba

    feature_bins[:, :, :, 0:2] = np.sort(feature_bins[:, :, :, 0:2], axis=2)
    feature_bins[:, :, :, 2:4] = np.sort(feature_bins[:, :, :, 2:4], axis=2)

    mn = np.min(feature_bins[:, :, :, 0:2], axis=3)
    comatrix_bins = np.ones((feature_bins.shape[0], feature_bins.shape[1], feature_bins.shape[2]))
    for k in range(len(all_bins)):
        comatrix_bins += (feature_bins[:, :, :, k].astype(float) - 1) * np.prod(all_bins[0:k])

    comatrix_bins[mn < 0] = -1
    comatrix_bins = comatrix_bins.flatten()
    comatrix_bins = comatrix_bins[comatrix_bins > 0]

    return comatrix_bins


def __calc_angular_bins(aBins, x0, x1, y0, y1):
    if aBins < 2:
        ba = np.ones(x0.shape)
        return ba

    a0 = (x0 ** 2 + y0 ** 2) ** 0.5
    a1 = (x1 ** 2 + y1 ** 2) ** 0.5
    a01 = ((x0 - x1) ** 2 + (y0 - y1) ** 2) ** 0.5
    e = 1e-10
    cosa = (a1 ** 2 + a0 ** 2 - a01 ** 2) / 2. / (a1 + e) / (a0 + e)
    cosa[cosa > 1] = 1
    cosa[cosa < -1] = -1
    ba = np.floor(np.arccos(cosa) / np.pi * aBins).astype(np.int8) + 1
    ba[ba < 1] = 1
    ba[ba > aBins] = aBins

    return ba


def __crop_roi(b, bg, gx, gy, ii, jj, kk):
    b0 = b[ii[0]:ii[1], jj[0]:jj[1], kk[0]:kk[1]]
    g0 = bg[ii[0]:ii[1], jj[0]:jj[1], kk[0]:kk[1]]
    x0 = gx[ii[0]:ii[1], jj[0]:jj[1], kk[0]:kk[1]]
    y0 = gy[ii[0]:ii[1], jj[0]:jj[1], kk[0]:kk[1]]
    return b0, g0, x0, y0


def __crop_using_mask(im, mask, dists, z2xy):
    mask[mask < 0] = 0

    d = [1 + max(dists)] * 3
    d[2] = 1 + round(0.5 + max(dists) / z2xy)
    bounds = []
    for dim in range(len(im.shape)):
        projection = np.sum(mask, axis=dim).flatten()
        idx = np.argwhere(projection > 0)
        first = int(max(0, idx[0] - d[dim]))
        last = int(min(len(projection), idx[-1] + d[dim]))
        bounds.append((first, last))

    im = im[bounds[0][0]:bounds[0][1], bounds[1][0]:bounds[1][1], bounds[2][0]:bounds[2][1]]
    mask = mask[bounds[0][0]:bounds[0][1], bounds[1][0]:bounds[1][1], bounds[2][0]:bounds[2][1]]

    return im, mask


def __to_bins(feature, feature_range, feature_bins):
    binned = (feature - feature_range[0]) / float(feature_range[1] - feature_range[0])
    binned = (np.floor(binned * feature_bins) + 1).astype(np.int8)
    binned[binned < 1] = 1
    binned[binned > feature_bins] = feature_bins
    return binned


def __zucker_hummel_operator():
    s2 = 2**(-0.5)
    s3 = 3**(-0.5)

    operator = np.zeros((3, 3, 3), dtype=float)
    operator[:, :, 2] = np.asarray([[s3, s2, s3], [s2, 1, s2], [s3, s2, s3]])
    operator[:, :, 0] = -operator[:, :, 2]
    operator = operator / np.sum(np.abs(operator))

    return operator


def __calculate_gradients(gray_image3d, z2xy):
    operator = __zucker_hummel_operator()

    grad_x = correlate(gray_image3d, operator.swapaxes(0, 2), mode='same')
    grad_y = correlate(gray_image3d, operator.swapaxes(1, 2), mode='same')
    grad_z = correlate(gray_image3d, operator, mode='same') / z2xy

    return grad_x, grad_y, grad_z


def calc_offsets(dists, econ, z2xy):
    if not econ:
        return calc_offsets_all(dists, z2xy)
    else:
        return calc_offsets_econ(dists, z2xy)


def calc_offsets_econ(dists, z2xy):
    angles = [[0., np.pi / 2]]
    phi = 0
    for i in range(4):
        theta = np.pi / 4 * i
        angles.append([phi, theta])

    phi = np.pi / 4
    for i in range(8):
        theta = np.pi / 4 * i
        angles.append([phi, theta])

    offsets = []
    for d in dists:
        for angle in angles:
            theta = angle[0]
            phi = angle[1]
            x = round(d * np.cos(theta) * np.cos(phi))
            y = round(d * np.sin(theta) * np.cos(phi))
            z = round(0.4 + d * np.sin(phi) / z2xy)
            if not [d, x, y, z] in offsets:
                offsets.append([d, x, y, z])

    offsets.sort()
    offsets = np.asarray(offsets).astype(int)
    return offsets


def calc_offsets_all(dists, z2xy):
    max_dist = max(dists)
    offsets = []

    z = 0
    y = 0
    for x in range(1, max_dist + 1):
        __add_offset(offsets, dists, x, y, z, z2xy)

    z = 0
    for x in range(-max_dist, max_dist + 1):
        for y in range(1, max_dist + 1):
            __add_offset(offsets, dists, x, y, z, z2xy)

    for x in range(-max_dist, max_dist + 1):
        for y in range(-max_dist, max_dist + 1):
            for z in range(1, max_dist + 1):
                __add_offset(offsets, dists, x, y, z, z2xy)

    offsets.sort()
    offsets = np.asarray(offsets)
    return offsets


def __add_offset(offsets, dists, x, y, z, z2xy):
    d = round((x ** 2 + y ** 2 + (z * z2xy) ** 2) ** 0.5)
    if d in dists:
        offsets.append([d, x, y, z])


def main():
    sz = 64
    xx = np.linspace(0, 1, sz)
    const_grad, _, _ = np.meshgrid(xx, xx * 0, xx * 0)
    im1 = const_grad * 0.9 + np.random.rand(sz, sz, sz) * 0.1

    cm = cooccur3D(im1, i_bins=6, dists=(1,), econ=True)
    print(cm.astype(int))


if __name__ == '__main__':
    main()