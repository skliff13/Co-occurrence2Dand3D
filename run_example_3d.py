import nibabel as nb

from cooccur3D import cooccur3D


def example3d():
    nii = nb.load('test_data/CT_scan_with_segmented_lungs.nii.gz')
    im3 = nii.get_data().swapaxes(0, 1)

    mask = im3 > -1500

    i_range = (-900, 400)  # in Hounsfield Units (HU)

    voxel_dimensions = nii.header.get('pixdim')
    z2xy = voxel_dimensions[3] / voxel_dimensions[1]

    comatrix = cooccur3D(im3, mask=mask, i_bins=6, i_range=i_range, dists=(1, 2), z2xy=z2xy, econ=True)

    print('Matrix shape: (Num_Distances, A_bins, G_bins, G_bins, I_bins, I_bins)')
    print(comatrix.shape)
    print(comatrix.astype(int))

    return


if __name__ == '__main__':
    example3d()