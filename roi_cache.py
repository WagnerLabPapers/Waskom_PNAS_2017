from __future__ import print_function
import sys
import os
import os.path as op

import numpy as np
import pandas as pd
from scipy import interpolate, linalg
from scipy.spatial.distance import cdist, pdist, squareform

import nibabel as nib
import nibabel.freesurfer as fs
from nibabel.affines import apply_affine

from moss.external.mvpasurf import Surface

import lyman
from surfutils import epi_to_surf_xfm


PROJECT = lyman.gather_project_info()


def backproject_label(label_file, subj, hemi):
    """Return label indices on individual subject surface."""
    label_verts = fs.read_label(label_file)

    # Define map of label on fsaverage surface
    label = np.zeros(163842, np.int)
    label[label_verts] = 1

    # Reverse normalize and convert to vertex indices
    subj_label = surface_transform(label, subj, hemi)
    subj_label_verts = np.argwhere(subj_label).squeeze()
    return subj_label_verts


def surface_transform(vals, subj, hemi, direction="reverse"):
    """Transform a surface scalar map using spherical transform.

    Parameters
    ----------
    vals : array or Series
        Scalar value map to transform.
    subj : string
        Freesurfer subject ID.
    hemi : lh | rh
        Hemisphere data are defined on
    direction : reverse | forward
        Whether transformation should be from group space to subject space
        (reverse) or the other direction (forward).

    Returns
    -------
    out_vals : array or Series
        Scalar value map defined on new surface.

    """
    data_dir = PROJECT["data_dir"]
    sphere_reg_fname = op.join(data_dir, subj, "surf", hemi + ".sphere.reg")
    avg_sphere_fname = op.join(data_dir, "fsaverage/surf", hemi + ".sphere")

    sphere_reg, _ = nib.freesurfer.read_geometry(sphere_reg_fname)
    avg_sphere, _ = nib.freesurfer.read_geometry(avg_sphere_fname)

    if direction.startswith("f"):
        src_sphere, trg_sphere = sphere_reg, avg_sphere
    elif direction.startswith("r"):
        src_sphere, trg_sphere = avg_sphere, sphere_reg

    interpolator = interpolate.NearestNDInterpolator(src_sphere, vals)
    out_vals = interpolator(trg_sphere)

    if isinstance(vals, pd.Series):
        out_vals = pd.Series(out_vals)

    return out_vals


def surf_to_vol(surf, roi_verts, xfm):
    """Return voxels indices corresponding to verts.

    Paramters
    ---------
    surf : moss.external.mvpasurf.Surface
        Surface geometry.
    roi_verts : 1d numpy array
        Vertex indices correpsonding to ROI.
    xfm : 4 x 4 numpy array
        Composite linear transform from functional IJK to anatomical RAS.

    Returns
    -------
    vox_ijk : n x 3 array
        Coordinates of ROI voxels in functional volume.
    vox2vert : Series
        Series mapping voxel index to surface vertex index.

    """
    vert_ras = surf.vertices[roi_verts]

    # Find oversampled voxel coordinates of the ROI and reduce
    inv_xfm = linalg.inv(xfm)
    vox_ijk_all = apply_affine(inv_xfm, vert_ras).round().astype(np.int)
    vox_ijk = pd.DataFrame(vox_ijk_all).drop_duplicates().values

    # Map the coordinates back into surface space and find closest vertex
    vox_ras = apply_affine(xfm, vox_ijk)
    nearest_vert = cdist(vox_ras, vert_ras).argmin(axis=1)
    vert_idx = roi_verts[nearest_vert]
    vert_ras = surf.vertices[vert_idx]

    # Make mapping from vertex to voxels unique
    vert2vox_dist = pd.DataFrame(cdist(vert_ras, vox_ras),
                                 index=vert_idx)
    ordered_verts = vert2vox_dist.min(axis=1).argsort().index
    vert2vox = (vert2vox_dist.idxmin(axis=1)
                             .reindex(ordered_verts)
                             .drop_duplicates())

    # Reduce the voxel coordinates to voxels with a unique vertex pair
    vox_ijk = vox_ijk[np.sort(vert2vox.values)]

    # Build a mapping from new voxel indices to vetex indices
    vox2vert = pd.Series(vert2vox.index)

    return vox_ijk, vox2vert


def create_2D_distance_matrix(surf, vox2vert, maxdistance=50):
    """Compute distance between voxels along the cortical surface.

    Parameters
    ----------
    surf : moss.external.mvpasurf.Surface
        Surface geometry.
    vox2vert : Series
        Series mapping voxel index to surface vertex index.
    maxdistance : int, optional
        Maximum distance on the surface to consider.

    Returns
    -------
    dmat : array
        Square distance matrix, possibly with nans.

    """
    dmat = pd.DataFrame(index=vox2vert.index,
                        columns=vox2vert.index,
                        dtype=np.float)

    vert2vox = pd.Series(vox2vert.index, vox2vert.values)

    for seed_vox, seed_vert in vox2vert.iteritems():
        dvec = pd.Series(surf.dijkstra_distance(seed_vert, 50))
        dvec = dvec[dvec.index.isin(vert2vox.index)]
        dvec.index = pd.Series(dvec.index).map(vert2vox)
        dmat.ix[seed_vox].update(dvec)

    return dmat.values


def create_3D_distance_matrix(vox_ijk, epi_fname):
    """Compute distance between voxels in the volume.

    Parameters
    ----------
    vox_ijk : n x 3 array
        Indices of voxels included in the ROI.
    epi_fname : file path
        Path to image defining the volume space.

    Returns
    -------
    dmat : array
        Dense square distance matrix.

    """
    aff = nib.load(epi_fname).affine
    vox_ras = nib.affines.apply_affine(aff, vox_ijk)
    dmat = squareform(pdist(vox_ras))

    return dmat


def extract_from_volume(vol_data, vox_ijk):
    """Extract data values (broadcasting across time if relevant)."""
    i, j, k = vox_ijk.T
    ii, jj, kk = vol_data.shape[:3]
    fov = (np.in1d(i, np.arange(ii)) &
           np.in1d(j, np.arange(jj)) &
           np.in1d(k, np.arange(kk)))

    if len(vol_data.shape) == 3:
        ntp = 1
    else:
        ntp = vol_data.shape[-1]

    roi_data = np.empty((len(i), ntp))
    roi_data[:] = np.nan
    roi_data[fov] = vol_data[i[fov], j[fov], k[fov]]
    return roi_data


def prepare_hemisphere(exp, subj, hemi, roi):
    """Create relevant ROI-defining objects for one hemisphere."""
    label_fname = "roi_labels/{}.{}.label".format(hemi, roi)
    roi_verts = backproject_label(label_fname, subj, hemi)

    # Get paths to the image defining functional space
    # and the matrix encoding a transformation to anatomical space
    anal_dir = PROJECT["analysis_dir"]
    base_dir = op.join(anal_dir, exp, subj, "reg/epi/unsmoothed/run_1")
    epi_fname = op.join(base_dir, "mean_func_xfm.nii.gz")
    reg_fname = op.join(base_dir, "func2anat_tkreg.dat")
    xfm = epi_to_surf_xfm(epi_fname, reg_fname)

    # Load in the surface geometry
    data_dir = PROJECT["data_dir"]
    surf_fname = op.join(data_dir, subj, "surf", hemi + ".graymid")
    surf = Surface(*fs.read_geometry(surf_fname))

    # Identify voxel coordinates and mapping to the surface
    vox_ijk, vox2vert = surf_to_vol(surf, roi_verts, xfm)

    # Compute the voxel-to-voxel distances in 2 and 3 dimensions
    dmat2d = create_2D_distance_matrix(surf, vox2vert)
    dmat3d = create_3D_distance_matrix(vox_ijk, epi_fname)

    return vox_ijk, vox2vert, dmat2d, dmat3d


def extract_data(exp, subj, roi_info):
    """Extract timeseries data from each ROI."""
    ts_data = {roi: [] for roi in roi_info}
    n_runs = dict(dots=12, sticks=12, rest=8)[exp]

    anal_dir = PROJECT["analysis_dir"]
    ts_temp = op.join(anal_dir, exp, subj, "reg",  "epi", "unsmoothed",
                      "run_{}", "res4d_xfm.nii.gz")

    # Extract ROI data from each run, loading images only once
    for run in range(1, n_runs + 1):
        run_data = nib.load(ts_temp.format(run)).get_data()

        for roi, info in roi_info.iteritems():
            roi_ts = extract_from_volume(run_data, info["vox_ijk"])
            ts_data[roi].append(roi_ts)

    # Combine across runs
    ts_data = {roi: np.hstack(ts_data[roi]).T for roi in roi_info}
    for roi in roi_info:
        assert ts_data[roi].shape[1] == len(roi_info[roi]["vox_ijk"])

    return ts_data


if __name__ == "__main__":

    # Parse the arguments
    if len(sys.argv) < 3:
        sys.exit("Usage: roi_cache.py <subj> <exp> <roi> (<roi> ...)")
    else:
        subj = sys.argv[1]
        exp = sys.argv[2]
        rois = sys.argv[3:]

    # Ensure that the output exists
    if not op.exists("roi_cache"):
        os.mkdir("roi_cache")

    # Build necessary ROI data
    info = {}
    for roi in rois:

        roi_info = dict(vox_ijk=[], vox2vert=[], dmat2d=[], dmat3d=[])

        for hemi in ["lh", "rh"]:

            hemi_info = prepare_hemisphere(exp, subj, hemi, roi)
            vox_ijk, vox2vert, dmat2d, dmat3d = hemi_info
            roi_info["vox_ijk"].append(vox_ijk)
            roi_info["vox2vert"].append(vox2vert)
            roi_info["dmat2d"].append(dmat2d)
            roi_info["dmat3d"].append(dmat3d)

        # Combine data across hemispheres
        n_lh = len(roi_info["vox_ijk"][0])
        n_rh = len(roi_info["vox_ijk"][1])
        hemispheres = np.array(["lh"] * n_lh + ["rh"] * n_rh)
        vox_ijk = np.vstack([d for d in roi_info["vox_ijk"]])
        vox2vert = np.hstack([d for d in roi_info["vox2vert"]])
        ur = np.ones((n_lh, n_rh)) * np.nan
        ll = ur.T
        dmat2d = np.r_[np.c_[roi_info["dmat2d"][0], ur],
                       np.c_[ll, roi_info["dmat2d"][1]]]
        dmat3d = np.r_[np.c_[roi_info["dmat3d"][0], ur],
                       np.c_[ll, roi_info["dmat3d"][1]]]

        info[roi] = dict(hemispheres=hemispheres,
                         vox_ijk=vox_ijk,
                         vox2vert=vox2vert,
                         dmat2d=dmat2d,
                         dmat3d=dmat3d)

    # Extract timeseries data
    # This is written a little indirectly so we don't need to
    # load in each (large) timeseries image multiple times
    ts_data = extract_data(exp, subj, info)

    # Save the data to disk
    for roi in rois:
        fname = "roi_cache/{}_{}_{}.npz".format(subj, exp, roi)
        roi_info = info[roi]
        roi_info["ts_data"] = ts_data[roi]
        np.savez(fname, **roi_info)
