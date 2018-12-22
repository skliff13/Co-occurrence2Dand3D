# -*- coding: utf-8 -*-
import numpy as np
from skimage import io
from scipy.signal import correlate2d


def cooccur2D(m, iRange=(0, 1), iBins=8, gRange=(0, 1), gBins=1, aBins=1, dists=(1,)):
    m = m.astype(float)

    b = __to_bins(m, iRange, iBins)

    gx, gy = __calculate_gradients(m)
    gm = (gx**2 + gy**2)**0.5

    bg = __to_bins(gm, gRange, gBins)

    LUT = calcLUT(dists)

    bn = [iBins, iBins, gBins, gBins, aBins]
    prodbn = int(np.prod(bn))
    desc = np.zeros((len(dists) * prodbn))
    hh = [0] * LUT.shape[0]

    for i in range(LUT.shape[0]):
        b0 = b.copy()
        g0 = bg.copy()
        x0 = gx.copy()
        y0 = gy.copy()

        id = dists.index(LUT[i, 0])

        LUTr = LUT[LUT[:, 0] == dists[id], :].copy()
        LUTr = np.append(LUTr, [[0] * LUT.shape[1]], axis=0)

        r = (max(-LUTr[:, 2]), max(LUTr[:, 2]), max(-LUTr[:, 1]), max(LUTr[:, 1]))

        ii = (r[0], b.shape[0] - r[1])
        jj = (r[2], b.shape[1] - r[3])

        b0 = b0[ii[0]:ii[1], jj[0]:jj[1]].copy()
        g0 = g0[ii[0]:ii[1], jj[0]:jj[1]].copy()
        x0 = x0[ii[0]:ii[1], jj[0]:jj[1]].copy()
        y0 = y0[ii[0]:ii[1], jj[0]:jj[1]].copy()
        a0 = (x0**2 + y0**2)**0.5

        ii1 = (r[0] + LUT[i, 2], b.shape[0] - r[1] + LUT[i, 2])
        jj1 = (r[2] + LUT[i, 1], b.shape[1] - r[3] + LUT[i, 1])

        b1 = b[ii1[0]:ii1[1], jj1[0]:jj1[1]]
        g1 = bg[ii1[0]:ii1[1], jj1[0]:jj1[1]]

        x1 = gx[ii1[0]:ii1[1], jj1[0]:jj1[1]]
        y1 = gy[ii1[0]:ii1[1], jj1[0]:jj1[1]]
        a1 = (x1**2 + y1**2)**0.5

        a01 = ((x0 - x1)**2 + (y0 - y1)**2)**0.5
        e = 1e-10
        cosa = (a1**2 + a0**2 - a01**2) / 2. / (a1 + e) / (a0 + e)
        cosa[cosa > 1] = 1
        cosa[cosa < -1] = -1
        if gBins > 1:
            cosa[g1 == 1] = 1

        ba = np.floor(np.arccos(cosa) / np.pi * aBins).astype(np.int8) + 1
        ba[ba < 1] = 1
        ba[ba > aBins] = aBins

        cobins = np.zeros((b0.shape[0], b0.shape[1], 5)).astype(np.int8)
        cobins[:, :, 0] = b0
        cobins[:, :, 1] = b1
        cobins[:, :, 2] = g0
        cobins[:, :, 3] = g1
        cobins[:, :, 4] = ba

        cobins[:, :, 0:2] = np.sort(cobins[:, :, 0:2], axis=2)
        cobins[:, :, 2:4] = np.sort(cobins[:, :, 2:4], axis=2)
        mn = np.min(cobins[:, :, 0:2], axis=2)

        cumul = np.ones((cobins.shape[0], cobins.shape[1]))
        for k in range(len(bn)):
            cumul += (cobins[:, :, k].astype(float) - 1) * np.prod(bn[0:k])
        cumul[mn < 0] = -1

        cumul = cumul.flatten()
        cumul = cumul[cumul > 0]

        h, _ = np.histogram(cumul, bins=range(1, prodbn + 2, 1))
        h2 = np.zeros(desc.shape)
        h2[id * prodbn:(id + 1) * prodbn] = h
        desc += h2

    return desc


def __to_bins(feature, feature_range, feature_bins):
    b = (feature - feature_range[0]) / float(feature_range[1] - feature_range[0])
    b = (np.floor(b * feature_bins) + 1).astype(np.int8)
    b[b < 1] = 1
    b[b > feature_bins] = feature_bins
    return b


def __calculate_gradients(gray_image2d):
    sobel_operator = np.asarray([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    grad_x = correlate2d(gray_image2d, sobel_operator, mode='same', boundary='symm')
    grad_y = correlate2d(gray_image2d, np.transpose(sobel_operator), mode='same', boundary='symm')
    return grad_x, grad_y


def calcLUT(dists):

    def __add_LUT_element(LUT, dists, x, y):
        s3 = 3. ** 0.5
        d = round((x ** 2 + y ** 2) ** 0.5)
        if d in dists:
            x1 = round(x / 2. - s3 / 2. * y)
            y1 = round(y / 2. + s3 / 2. * x)
            LUT.append([d, x, y, x1, y1])

    maxd = max(dists)
    LUT = []

    y = 0
    for x in range(1, maxd + 1):
        __add_LUT_element(LUT, dists, x, y)

    for x in range(-maxd, maxd + 1):
        for y in range(1, maxd + 1):
            __add_LUT_element(LUT, dists, x, y)

    LUT.sort()
    LUT = np.asarray(LUT)
    return LUT


def main():
    im = io.imread('lena256_gray.png').astype(float)
    if np.max(im) > 1:
        im /= 255.

    cm = cooccur2D(im, iBins=6, dists=(1,))
    print(cm.astype(int))


if __name__ == '__main__':
    main()