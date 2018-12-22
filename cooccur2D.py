# -*- coding: utf-8 -*-
import numpy as np
from skimage import io
from scipy.signal import correlate2d


def cooccur2D(gray_image2d, i_range=(0, 1), i_bins=8, g_range=(0, 1), g_bins=1, a_bins=1, dists=(1,), mask=None):
    gray_image2d = gray_image2d.astype(float)

    if mask is not None:
        gray_image2d, mask = __crop_using_mask(gray_image2d, mask)

    binned_i = __to_bins(gray_image2d, i_range, i_bins)
    if mask is not None:
        binned_i[mask == 0] = -1

    grad_x, grad_y = __calculate_gradients(gray_image2d)
    grad_magnitude = (grad_x**2 + grad_y**2)**0.5

    binned_g = __to_bins(grad_magnitude, g_range, g_bins)

    offsets = calc_offsets(dists)

    all_bins = [i_bins, i_bins, g_bins, g_bins, a_bins]
    bins_prod = int(np.prod(all_bins))
    result = np.zeros((len(dists) * bins_prod))

    for i in range(offsets.shape[0]):
        dist_idx = dists.index(offsets[i, 0])

        dist_offs = offsets[offsets[:, 0] == dists[dist_idx], :].copy()
        dist_offs = np.append(dist_offs, [[0] * offsets.shape[1]], axis=0)

        to_crop = (max(-dist_offs[:, 2]), max(dist_offs[:, 2]), max(-dist_offs[:, 1]), max(dist_offs[:, 1]))

        ii = (to_crop[0], binned_i.shape[0] - to_crop[1])
        jj = (to_crop[2], binned_i.shape[1] - to_crop[3])
        b0, g0, x0, y0 = __crop_roi(binned_i, binned_g, grad_x, grad_y, ii, jj)

        ii1 = (to_crop[0] + offsets[i, 2], binned_i.shape[0] - to_crop[1] + offsets[i, 2])
        jj1 = (to_crop[2] + offsets[i, 1], binned_i.shape[1] - to_crop[3] + offsets[i, 1])
        b1, g1, x1, y1 = __crop_roi(binned_i, binned_g, grad_x, grad_y, ii1, jj1)

        ba = __calc_angular_bins(a_bins, x0, x1, y0, y1)

        comatrix_bins = __map_matrix_bins(all_bins, b0, b1, ba, g0, g1)

        h, _ = np.histogram(comatrix_bins, bins=range(1, bins_prod + 2, 1))
        h2 = np.zeros(result.shape)
        h2[dist_idx * bins_prod:(dist_idx + 1) * bins_prod] = h
        result += h2

    all_dims = all_bins.copy()
    all_dims.append(len(dists))
    dims = []
    for dim in all_dims:
        if dim > 1:
            dims.append(dim)
    dims.reverse()

    result = result.reshape(tuple(dims))

    return result


def __map_matrix_bins(all_bins, b0, b1, ba, g0, g1):
    feature_bins = np.zeros((b0.shape[0], b0.shape[1], 5)).astype(np.int8)
    feature_bins[:, :, 0] = b0
    feature_bins[:, :, 1] = b1
    feature_bins[:, :, 2] = g0
    feature_bins[:, :, 3] = g1
    feature_bins[:, :, 4] = ba

    feature_bins[:, :, 0:2] = np.sort(feature_bins[:, :, 0:2], axis=2)
    feature_bins[:, :, 2:4] = np.sort(feature_bins[:, :, 2:4], axis=2)

    mn = np.min(feature_bins[:, :, 0:2], axis=2)
    comatrix_bins = np.ones((feature_bins.shape[0], feature_bins.shape[1]))
    for k in range(len(all_bins)):
        comatrix_bins += (feature_bins[:, :, k].astype(float) - 1) * np.prod(all_bins[0:k])

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


def __crop_roi(b, bg, gx, gy, ii, jj):
    b0 = b[ii[0]:ii[1], jj[0]:jj[1]]
    g0 = bg[ii[0]:ii[1], jj[0]:jj[1]]
    x0 = gx[ii[0]:ii[1], jj[0]:jj[1]]
    y0 = gy[ii[0]:ii[1], jj[0]:jj[1]]
    return b0, g0, x0, y0


def __crop_using_mask(im, mask):
    mask[mask < 0] = 0

    bounds = []
    for dim in range(len(im.shape)):
        projection = np.sum(mask, axis=dim).flatten()
        idx = np.argwhere(projection > 0)
        first = max(0, idx[0] - 1)
        last = min(len(projection), idx[-1] + 1)
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


def calc_offsets(dists):
    max_dist = max(dists)
    offsets = []

    y = 0
    for x in range(1, max_dist + 1):
        __add_offset(offsets, dists, x, y)

    for x in range(-max_dist, max_dist + 1):
        for y in range(1, max_dist + 1):
            __add_offset(offsets, dists, x, y)

    offsets.sort()
    offsets = np.asarray(offsets)
    return offsets


def __add_offset(offsets, dists, x, y):
    s3 = 3. ** 0.5
    d = round((x ** 2 + y ** 2) ** 0.5)
    if d in dists:
        x1 = round(x / 2. - s3 / 2. * y)
        y1 = round(y / 2. + s3 / 2. * x)
        offsets.append([d, x, y, x1, y1])


def main():
    im = io.imread('lena256_gray.png').astype(float)
    if np.max(im) > 1:
        im /= 255.

    cm = cooccur2D(im, i_bins=6, dists=(1,))
    print(cm.astype(int))


if __name__ == '__main__':
    main()