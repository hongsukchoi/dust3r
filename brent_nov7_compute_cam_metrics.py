import pickle
from pathlib import Path
from typing import Any, Literal, Tuple, Dict, Union
from typing_extensions import TypeAlias

import numpy as np
import tyro
import viser.transforms as vtf
from tqdm import tqdm


def compute_similarity_transform(
    points_y: np.ndarray,
    points_x: np.ndarray,
    fix_scale: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute similarity transform parameters using the Umeyama method.

    Minimizes:
        mean( || Y - s * (R @ X) + t ||^2 )
    with respect to s, R, and t.

    Args:
        points_y: Array of shape (*batch, N, 3)
        points_x: Array of shape (*batch, N, 3)
        fix_scale: Whether to fix scale to 1

    Returns:
        Tuple of (s, R, t) where:
            s: Scale factor of shape (*batch, 1)
            R: Rotation matrix of shape (*batch, 3, 3)
            t: Translation vector of shape (*batch, 3)
    """
    *dims, N, _ = points_y.shape

    # subtract mean
    my = np.mean(points_y, axis=-2)  # (*, 3)
    mx = np.mean(points_x, axis=-2)
    y0 = points_y - my[..., None, :]  # (*, N, 3)
    x0 = points_x - mx[..., None, :]

    # correlation
    C = np.matmul(y0.swapaxes(-1, -2), x0) / N  # (*, 3, 3)
    U, D, Vh = np.linalg.svd(C)  # (*, 3, 3), (*, 3), (*, 3, 3)

    # handle reflection case
    S = np.eye(3).reshape(*(1,) * len(dims), 3, 3)
    S = np.broadcast_to(S, (*dims, 3, 3)).copy()

    neg = np.logical_and(
        np.linalg.det(U) * np.linalg.det(Vh.swapaxes(-1, -2)) < 0, True
    )
    S[neg] = S[neg] * np.diag([1, 1, -1])

    R = np.matmul(U, np.matmul(S, Vh))  # (*, 3, 3)

    if fix_scale:
        s = np.ones((*dims, 1), dtype=np.float32)
    else:
        var = np.sum(np.square(x0), axis=(-1, -2), keepdims=True) / N  # (*, 1, 1)
        # Create diagonal matrices for each batch dimension
        D_matrices = np.zeros((*dims, 3, 3))
        for idx in np.ndindex(*dims):
            D_matrices[idx] = np.diag(D[idx])

        DS = np.matmul(D_matrices, S)
        s = (
            np.sum(np.diagonal(DS, axis1=-2, axis2=-1), axis=-1, keepdims=True)
            / var[..., 0]
        )  # (*, 1)

    t = my - s * np.matmul(R, mx[..., None])[..., 0]  # (*, 3)

    assert s.shape == (*dims, 1)
    assert R.shape == (*dims, 3, 3)
    assert t.shape == (*dims, 3)

    return s, R, t


def align_points(
    points_x: np.ndarray,
    points_y: np.ndarray,
    fix_scale: bool = False,
) -> np.ndarray:
    """Align points_x to points_y using similarity transform.

    Args:
        points_x: Array of shape (*batch, N, 3) to be aligned
        points_y: Array of shape (*batch, N, 3) target points
        fix_scale: Whether to fix scale to 1

    Returns:
        Aligned points of shape (*batch, N, 3)
    """
    s, R, t = compute_similarity_transform(points_y, points_x, fix_scale)

    aligned_x = (
        s[..., None, :] * np.einsum("...ij,...nj->...ni", R, points_x) + t[..., None, :]
    )
    assert aligned_x.shape == points_x.shape
    return aligned_x


def compute_relpose_metrics(
    gt_world_cameras: Dict[str, Any],
    our_pred_world_cameras_and_structure: Dict[str, Any],
    align_scale: bool,
) -> Union[Dict[str, float], Literal["found nans"]]:
    out = {}

    # What metrics are used in RelPose++?
    # - Joint Rotation Accuracy @ 15 deg
    # - Camera Center Accuracy @ 0.2 (20% of scene scale)
    #     - Aligned using similarity transform
    assert gt_world_cameras.keys() == our_pred_world_cameras_and_structure.keys()
    cam_keys = tuple(gt_world_cameras.keys())
    assert "cam2world" in gt_world_cameras[cam_keys[0]]
    assert "cam2world" in gt_world_cameras[cam_keys[0]]
    assert our_pred_world_cameras_and_structure[cam_keys[0]]["cam2world"].shape == (4, 4)

    gt_cam_positions = np.array(
        [gt_world_cameras[cam_key]["cam2world"][:3, 3] for cam_key in cam_keys]
    )
    pred_cam_positions = np.array(
        [our_pred_world_cameras_and_structure[cam_key]["cam2world"][:3, 3] for cam_key in cam_keys]
    )
    assert gt_cam_positions.shape == pred_cam_positions.shape == (len(cam_keys), 3)

    # Assumptions:
    # 1. The global coordinate system is arbitrary.
    # 2. Both the ground-truth and the predicted cameras are already in meters.
    # from utils import compute_similarity_transform

    if np.any(np.isnan(pred_cam_positions)):
        # breakpoint()
        return "found nans"

    s, R, t = compute_similarity_transform(
        points_x=pred_cam_positions,
        points_y=gt_cam_positions,
        fix_scale=not align_scale,  # True,
    )
    if align_scale:
        assert not np.isclose(s, 1.0)
    else:
        assert np.isclose(s, 1.0)

    pred_cam_positions = s * np.einsum("ij,nj->ni", R, pred_cam_positions) + t
    assert gt_cam_positions.shape == pred_cam_positions.shape == (len(cam_keys), 3)

    out["per_cam_pos_error"] = np.linalg.norm(
        (gt_cam_positions - pred_cam_positions), axis=-1
    )
    # out["all_cam_pos_mse"] = np.mean((gt_cam_positions - pred_cam_positions) ** 2)

    # Orientation accuracy.
    gt_cam_orientations = np.array(
        [gt_world_cameras[cam_key]["cam2world"][:3, :3] for cam_key in cam_keys]
    )
    gt_cam_orientations /= np.cbrt(np.linalg.det(gt_cam_orientations)[..., None, None])
    pred_cam_orientations = np.array(
        [our_pred_world_cameras_and_structure[cam_key]["cam2world"][:3, :3] for cam_key in cam_keys]
    )
    pred_cam_orientations = np.einsum("ij,njk->nik", R, pred_cam_orientations)
    assert (
        gt_cam_orientations.shape
        == pred_cam_orientations.shape
        == (len(cam_keys), 3, 3)
    )

    # How to add a coordinate frame in viser:
    # Add all of the cameras.
    # for cam_key, cam_position, cam_orientation in zip(
    #     cam_keys, pred_cam_positions, pred_cam_orientations
    # ):
    #     server.scene.add_frame(
    #         f"/{cam_key}",
    #         show_axes=True,
    #         wxyz=vtf.SO3.from_matrix(cam_orientation).wxyz,
    #         position=cam_position,
    #     )
    #     server.scene.add_label(f"/{cam_key}/label", f"{cam_key[3:]}")

    # for cam_key, cam_position, cam_orientation in zip(
    #     cam_keys, gt_cam_positions, gt_cam_orientations
    # ):
    #     print(np.linalg.det(cam_orientation))
    # print(cam_orientation, cam_position)
    # server.scene.add_frame(
    #     f"/gt_{cam_key}",
    #     show_axes=True,
    #     axes_length=0.125,
    #     origin_radius=0.0,
    #     wxyz=vtf.SO3.from_matrix(cam_orientation).wxyz,
    #     position=cam_position,
    # )
    # server.scene.add_label(f"/gt_{cam_key}/label", f"{cam_key[3:]}")

    gt_cam_pairwise_orientations = np.einsum(
        "mij,nkj->mnik", gt_cam_orientations, gt_cam_orientations
    )
    pred_cam_pairwise_orientations = np.einsum(
        # "mij,nkj->mnik", gt_cam_orientations, gt_cam_orientations
        "mij,nkj->mnik",
        pred_cam_orientations,
        pred_cam_orientations,
        # "mij,nkj->mnik", pred_cam_orientations, pred_cam_orientations
    )

    pairwise_deltas = np.einsum(
        "mnij,mnkj->mnik", gt_cam_pairwise_orientations, pred_cam_pairwise_orientations
    )
    pairwise_deltas_radians = np.linalg.norm(
        vtf.SO3.from_matrix(pairwise_deltas).log(), axis=-1
    )
    assert pairwise_deltas_radians.shape == (len(cam_keys), len(cam_keys))

    # Get upper-triangular terms, not including the k=0 diagonal.
    # This is because the diagonal is always 0.
    upper_triangular_mask = np.triu(
        np.ones_like(pairwise_deltas_radians, dtype=bool), k=1
    )
    pairwise_deltas_radians = pairwise_deltas_radians[upper_triangular_mask]
    out["per_pair_deg_error"] = np.degrees(pairwise_deltas_radians)

    return out


def main(pkl_root_dir: Path, align_scale: bool = False) -> None:
    """Compute camera metrics for all pkl files in the given directory.

    Args:
        pkl_root_dir: Directory containing pkl files. We will glob for all pkl
            files that match "{pkl_root_dir}/*.pkl".
        align_scale: Whether to align scale of ground-truth and estimates.
    """
    pkl_paths = list(pkl_root_dir.glob("*.pkl"))

    orientation_errors_deg = []
    position_errors_m = []
    for pkl_path in tqdm(pkl_paths):
        with open(pkl_path, "rb") as f:
            result = pickle.load(f)
        assert result.keys() == {
            "gt_world_cameras",
            "gt_world_humans_smpl_params",
            "gt_world_structure",
            "our_pred_world_cameras_and_structure",
            "our_pred_humans_smplx_params",
            "hmr2_pred_humans_and_cameras",
            "dust3r_pred_world_cameras_and_structure",
        }

        metrics = compute_relpose_metrics(
            result["gt_world_cameras"],
            result["our_pred_world_cameras_and_structure"],
            align_scale=align_scale
        )
        if metrics == "found nans":
            print(f"Found nans in {pkl_path}")
        else:
            orientation_errors_deg.append(metrics["per_pair_deg_error"])
            position_errors_m.append(metrics["per_cam_pos_error"])

    orientation_errors_deg = np.concatenate(orientation_errors_deg)
    position_errors_m = np.concatenate(position_errors_m)
    assert orientation_errors_deg.ndim == position_errors_m.ndim == 1

    # Print average errors and stds.
    print(
        f"Orientation error (deg): {np.mean(orientation_errors_deg):.2f} ± {np.std(orientation_errors_deg) / np.sqrt(len(orientation_errors_deg)):.2f}"
    )
    print(
        f"Position error (m): {np.mean(position_errors_m):.2f} ± {np.std(position_errors_m) / np.sqrt(len(position_errors_m)):.2f}"
    )

    for deg_threshold in (5, 10, 15, 20, 25, 30):
        print(
            f"Orientation error < {deg_threshold} deg: {100.0 * np.mean(orientation_errors_deg < deg_threshold):.2f}"
        )


if __name__ == "__main__":
    tyro.cli(main)
