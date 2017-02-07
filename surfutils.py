import os.path as op
import numpy as np
import pandas as pd
import nibabel as nib
from nibabel.affines import apply_affine

import lyman


PROJECT = lyman.gather_project_info()


def roi_to_surf(exp, subj, vals, vox_ijk):
    """Transform a vector of ROI data onto the surface."""
    # Get paths to files defining the volumetric space
    analysis_dir = PROJECT["analysis_dir"]
    base_dir = op.join(analysis_dir, exp, subj, "reg/epi/unsmoothed/run_1")
    epi_fname = op.join(base_dir, "mean_func_xfm.nii.gz")
    reg_fname = op.join(base_dir, "func2anat_tkreg.dat")

    # Obtain a transform from the anatomy to the EPI space
    xfm = np.linalg.inv(epi_to_surf_xfm(epi_fname, reg_fname))

    # Put the ROI data into the volume
    epi_img = nib.load(epi_fname)
    vol = np.empty(epi_img.shape)
    vol[:] = np.nan
    i, j, k = vox_ijk.T
    vol[i, j, k] = vals

    # Sample the data onto the surface a hemisphere at a time
    surf_data = {}
    for hemi in ["lh", "rh"]:

        # Obtain voxel indices of surface coordinates
        i, j, k = surf_to_voxel_coords(subj, hemi, xfm)

        # Limit to the FOV of the acquisition
        ii, jj, kk = vol.shape
        fov = (np.in1d(i, np.arange(ii)) &
               np.in1d(j, np.arange(jj)) &
               np.in1d(k, np.arange(kk)))

        # Transform from volume to surface
        hemi_data = np.empty(len(i))
        hemi_data[:] = np.nan
        hemi_data[fov] = vol[i[fov], j[fov], k[fov]]
        surf_data[hemi] = pd.Series(hemi_data)

    # Combine across hemispheres and return
    surf_data = pd.concat(surf_data, names=["hemi", "vertex"])
    return surf_data


def epi_to_surf_xfm(epi_fname, reg_fname):
    """Obtain a transformation from epi voxels -> Freesurfer surf coords.

    Parameters
    ----------
    epi_fname : string
        Filename pointing at image defining the epi space.
    reg_fname : string
        Filename pointing at registration file (from bbregister) that maps
        ``epi_img_fname`` to the Freesurfer anatomy.

    Returns
    -------
    xfm : 4 x 4 numpy array
        Transformation matrix that can be applied to surf coords.

    """
    # Load the Freesurfer "tkreg" style transform file
    # Confusingly, this file actually encodes the anat-to-func transform
    anat2func_xfm = np.genfromtxt(reg_fname, skip_header=4, skip_footer=1)
    func2anat_xfm = np.linalg.inv(anat2func_xfm)

    # Get a tkreg-compatibile mapping from IJK to RAS
    epi_img = nib.load(epi_fname)
    mgh_img = nib.MGHImage(np.zeros(epi_img.shape[:3]),
                           epi_img.get_affine(),
                           epi_img.get_header())
    vox2ras_tkr = mgh_img.get_header().get_vox2ras_tkr()

    # Combine the two transformations
    xfm = np.dot(func2anat_xfm, vox2ras_tkr)

    return xfm


def surf_to_voxel_coords(subj, hemi, xfm, surf="graymid"):
    """Obtain voxel coordinates of surface vertices in the EPI volume.

    Parameters
    ----------
    subj : string
        Freesurfer subject ID.
    hemi : lh | rh
        Hemisphere of surface to map.
    xfm : 4 x 4 array
        Linear transformation matrix between spaces.
    surf : string
        Freesurfer surface name defining coords.

    Returns
    i, j, k : 1d int arrays
        Arrays of voxel indices.

    """
    # Load the surface geometry
    data_dir = PROJECT["data_dir"]
    surf_fname = op.join(data_dir, subj, "surf", "{}.{}".format(hemi, surf))
    coords, _ = nib.freesurfer.read_geometry(surf_fname)
    return apply_affine(xfm, coords).round().astype(np.int).T


