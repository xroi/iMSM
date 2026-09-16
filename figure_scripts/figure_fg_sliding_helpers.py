import os
import sys
from pathlib import Path

_fig_dir: Path = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
_repo_root: Path = _fig_dir.parent if _fig_dir.name == "figure_scripts" else _fig_dir
for _p in [str(_fig_dir), str(_repo_root)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

if not os.path.exists("data") and (_repo_root / "data").exists():
    os.chdir(_repo_root)
import pickle
import re
import importlib
import matplotlib.patheffects as pe
import matplotlib.path as mpath
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import MaxNLocator
import mdtraj as md
import numpy as np
from scipy.interpolate import CubicSpline

from iMSM.data_classes.input_data_classes import iMSMInput, iMSMInputSingleSim
from iMSM.data_classes.config_data_class import iMSMConfig
from iMSM.data_classes.msm_data_classes import iMSMMSM
from iMSM._1_categorize import default_categorize
from iMSM._2_embed import default_embed
from iMSM._4_msm import default_msm
from iMSM.extensions.npc.npc_utils import infinitesimal_generator
from iMSM.main import run


def create_wedge_marker(theta1_deg: float, theta2_deg: float) -> mpath.Path:
    """Create a unit-radius pie wedge marker path with identical scaling to circular scatter markers."""
    theta1_rad: float = float(np.radians(theta1_deg))
    theta2_rad: float = float(np.radians(theta2_deg))
    n_pts: int = max(int(np.abs(theta2_deg - theta1_deg) / 3.0), 16)
    angles: np.ndarray = np.linspace(theta1_rad, theta2_rad, n_pts)

    x: np.ndarray = np.cos(angles)
    y: np.ndarray = np.sin(angles)

    verts: list[tuple[float, float]] = [
        (-1.0, -1.0),
        (1.0, -1.0),
        (1.0, 1.0),
        (-1.0, 1.0),
        (0.0, 0.0),
    ] + [(float(xi), float(yi)) for xi, yi in zip(x, y)] + [(0.0, 0.0)]

    codes: list[int] = (
        [mpath.Path.MOVETO, mpath.Path.MOVETO, mpath.Path.MOVETO, mpath.Path.MOVETO]
        + [mpath.Path.MOVETO]
        + [mpath.Path.LINETO] * len(angles)
        + [mpath.Path.CLOSEPOLY]
    )
    return mpath.Path(verts, codes)


def create_imsm_input(
    kap_coords: np.ndarray,
    focal_coords: np.ndarray,
    focal_component_name: str,
) -> iMSMInput:
    """Construct an iMSMInput instance for Kap95 and a given focal component."""
    full_coords: np.ndarray = np.concatenate((kap_coords, focal_coords), axis=0)
    components: np.ndarray = np.array(
        [f"kapC{i}" for i in range(kap_coords.shape[0])] + [focal_component_name]
    )
    trajectories: list[iMSMInputSingleSim] = [iMSMInputSingleSim(trajectory=full_coords)]
    return iMSMInput(
        trajectories=trajectories,
        components=components,
        focal_component=focal_component_name,
    )


def create_imsm_config(
    checkpoints_path: str,
    window_size: int,
    interaction_capacity: int,
    max_surface_dist: float,
    n_clusters: int,
    merge_cluster_threshold: float,
    tm_prior: float,
    start_stage: int,
    end_stage: int,
) -> iMSMConfig:
    """Create an iMSMConfig instance with explicit parameters."""
    config: iMSMConfig = iMSMConfig(checkpoints_path=checkpoints_path)
    config.window_size = window_size
    config.interaction_capacity = interaction_capacity
    config.max_surface_dist = max_surface_dist
    config.n_clusters = n_clusters
    config.merge_cluster_threshold = merge_cluster_threshold
    config.tm_prior = tm_prior
    config.start_stage = start_stage
    config.end_stage = end_stage
    return config


def setup_fg_sliding_data(
    traj_paths: list[str],
    top_path: str,
    imsm_checkpoint_base_path: str,
    stride: int,
    first_frame: int | None,
    last_frame: int | None,
) -> dict[str, object]:
    """Load trajectory, extract Kap95 and FG slices, and create iMSM inputs for single beads and COM."""
    traj: md.Trajectory = md.load(traj_paths, top=top_path)
    all_ca_indices: np.ndarray = traj.topology.select("name CA")

    kap_ca_indices: np.ndarray = all_ca_indices[:861]
    fg1_ca_indices: np.ndarray = all_ca_indices[861:986]
    fg2_ca_indices: np.ndarray = all_ca_indices[986:]
    fg_ca_indices: np.ndarray = fg1_ca_indices

    kap_xyz_mdtraj: np.ndarray = traj.xyz[:, kap_ca_indices, :]
    fg_xyz_mdtraj: np.ndarray = traj.xyz[:, fg_ca_indices, :]

    kap_coords: np.ndarray = np.transpose(kap_xyz_mdtraj, (1, 2, 0))
    fg_coords: np.ndarray = np.transpose(fg_xyz_mdtraj, (1, 2, 0))

    frame_slice: slice = slice(first_frame, last_frame, stride)
    kap_coords = kap_coords[:, :, frame_slice]
    fg_coords = fg_coords[:, :, frame_slice]

    c88_fg_coords: np.ndarray = fg_coords[88:89, :, :]
    c90_fg_coords: np.ndarray = fg_coords[90:91, :, :]
    com_88_91_fg_coords: np.ndarray = np.mean(fg_coords[88:92, :, :], axis=0, keepdims=True)

    c88_input: iMSMInput = create_imsm_input(
        kap_coords=kap_coords,
        focal_coords=c88_fg_coords,
        focal_component_name="fgC88",
    )
    c90_input: iMSMInput = create_imsm_input(
        kap_coords=kap_coords,
        focal_coords=c90_fg_coords,
        focal_component_name="fgC90",
    )
    com_88_91_input: iMSMInput = create_imsm_input(
        kap_coords=kap_coords,
        focal_coords=com_88_91_fg_coords,
        focal_component_name="fgC88_91_com",
    )

    imsm_runs: list[dict[str, object]] = [
        {
            "name": "fgC88",
            "input": c88_input,
            "checkpoint_path": os.path.join(imsm_checkpoint_base_path, "fgC88"),
            "focal_fg_alphacarbon": 88,
        },
        {
            "name": "fgC90",
            "input": c90_input,
            "checkpoint_path": os.path.join(imsm_checkpoint_base_path, "fgC90"),
            "focal_fg_alphacarbon": 90,
        },
        {
            "name": "fgC88_91_com",
            "input": com_88_91_input,
            "checkpoint_path": os.path.join(imsm_checkpoint_base_path, "fgC88_91_com"),
            "focal_fg_alphacarbon": (88, 91),
        },
    ]

    return {
        "traj": traj,
        "all_ca_indices": all_ca_indices,
        "kap_ca_indices": kap_ca_indices,
        "fg1_ca_indices": fg1_ca_indices,
        "fg2_ca_indices": fg2_ca_indices,
        "fg_ca_indices": fg_ca_indices,
        "kap_coords": kap_coords,
        "fg_coords": fg_coords,
        "imsm_runs": imsm_runs,
    }


def setup_fg_sliding_params(
    window_size: int,
    interaction_capacity: int,
    max_surface_dist: float,
    n_clusters: int,
    merge_cluster_threshold: float,
    tm_prior: float,
    start_stage: int,
    end_stage: int,
    ns_per_frame_base: float,
    stride: int,
) -> dict[str, object]:
    """Construct configuration dictionary for iMSM analysis."""
    ns_per_frame: float = ns_per_frame_base * stride
    return {
        "window_size": window_size,
        "interaction_capacity": interaction_capacity,
        "max_surface_dist": max_surface_dist,
        "n_clusters": n_clusters,
        "merge_cluster_threshold": merge_cluster_threshold,
        "tm_prior": tm_prior,
        "start_stage": start_stage,
        "end_stage": end_stage,
        "ns_per_frame": ns_per_frame,
        "stride": stride,
    }


def run_fg_sliding_imsm(
    imsm_runs: list[dict[str, object]],
    params: dict[str, object],
    top_path: str,
    kap_n_ca: int,
    heat5_range: tuple[int, int],
    heat6_range: tuple[int, int],
) -> None:
    """Execute iMSM calculation and reorder Markov states along the Y coordinate for all specified runs."""
    for item in imsm_runs:
        run_name: str = str(item["name"])
        run_input: iMSMInput = item["input"]  # type: ignore[assignment]
        run_chk_path: str = str(item["checkpoint_path"])
        print(f"\n================ Running iMSM for {run_name} ================")
        run_config: iMSMConfig = create_imsm_config(
            checkpoints_path=f"{run_chk_path}/",
            window_size=int(params["window_size"]),  # type: ignore[arg-type]
            interaction_capacity=int(params["interaction_capacity"]),  # type: ignore[arg-type]
            max_surface_dist=float(params["max_surface_dist"]),  # type: ignore[arg-type]
            n_clusters=int(params["n_clusters"]),  # type: ignore[arg-type]
            merge_cluster_threshold=float(params["merge_cluster_threshold"]),  # type: ignore[arg-type]
            tm_prior=float(params["tm_prior"]),  # type: ignore[arg-type]
            start_stage=int(params["start_stage"]),  # type: ignore[arg-type]
            end_stage=int(params["end_stage"]),  # type: ignore[arg-type]
        )
        run(input=run_input, config=run_config)
        reorder_imsm_states_by_y(
            imsm_checkpoint_path=run_chk_path,
            top_path=top_path,
            kap_n_ca=kap_n_ca,
            heat5_range=heat5_range,
            heat6_range=heat6_range,
            tm_prior=float(params["tm_prior"]),  # type: ignore[arg-type]
        )


def visualize_cluster_compositions(
    imsm_checkpoint_path: str,
    kap_n_ca: int,
    focal_component: str,
) -> None:
    """Plot cluster compositions heatmap for each Markov state."""
    with open(f"{imsm_checkpoint_path}/3_clustering.pickle", "rb") as f:
        imsm_cluster = pickle.load(f)
    raw_cluster_compositions: np.ndarray = np.array(imsm_cluster.clustering.cluster_centers_)
    comps: np.ndarray = np.array(imsm_cluster.unique_components)

    ordered_compositions: np.ndarray = np.zeros(
        (raw_cluster_compositions.shape[0], kap_n_ca + 1), dtype=float
    )
    for col_idx, comp_name in enumerate(comps):
        if str(comp_name).startswith("kapC"):
            res_idx: int = int(str(comp_name)[4:])
            if res_idx < kap_n_ca:
                ordered_compositions[:, res_idx] = raw_cluster_compositions[:, col_idx]
        elif str(comp_name) == "Unbound":
            ordered_compositions[:, -1] = raw_cluster_compositions[:, col_idx]

    plt.imshow(ordered_compositions, cmap="viridis", aspect="auto")
    plt.colorbar()
    plt.title(f"Cluster Compositions for focal component: {focal_component}")
    plt.xlabel("Component (alpha carbon index, last column=Unbound)")
    plt.ylabel("Cluster")
    plt.clim(0, 0.3)
    plt.show()


def visualize_transition_matrix(
    imsm_checkpoint_path: str,
    focal_component: str,
) -> None:
    """Plot the transition matrix heatmap from the MSM checkpoint."""
    with open(f"{imsm_checkpoint_path}/4_msm.pickle", "rb") as f:
        imsm_msm = pickle.load(f)
    msm_matrix: np.ndarray = imsm_msm.transition_matrix
    plt.imshow(msm_matrix, cmap="viridis")
    plt.colorbar()
    plt.title(f"Transition Matrix for focal component: {focal_component}")
    plt.xlabel("To State")
    plt.ylabel("From State")
    plt.show()


def print_state_makeup(
    title: str,
    histogram: np.ndarray,
    components: np.ndarray,
    top_n: int,
) -> None:
    """Print the top N components in a state histogram."""
    sorted_indices: np.ndarray = np.argsort(histogram)[::-1]
    print(f"\n--- {title} ---")
    count: int = 0
    for idx in sorted_indices:
        val: float = float(histogram[idx])
        if val <= 0.0 and count >= 1:
            break
        print(f"  {components[idx]:<12}: {val * 100.0:6.2f}%")
        count += 1
        if count >= top_n:
            break


def extract_focal_info(
    focal_fg_alphacarbon: int | tuple[int, int] | list[int] | str,
) -> tuple[list[int], str]:
    """Extract 0-indexed FG CA indices and focal component name."""
    if isinstance(focal_fg_alphacarbon, tuple):
        start_idx: int = int(focal_fg_alphacarbon[0])
        end_idx: int = int(focal_fg_alphacarbon[1])
        indices: list[int] = list(range(start_idx, end_idx + 1))
        name: str = f"fgC{start_idx}_{end_idx}_com"
        return indices, name

    if isinstance(focal_fg_alphacarbon, list):
        indices = [int(x) for x in focal_fg_alphacarbon]
        name = f"fgC{min(indices)}_{max(indices)}_com"
        return indices, name

    if isinstance(focal_fg_alphacarbon, str):
        match = re.search(r"(\d+)[-_](\d+)", focal_fg_alphacarbon)
        if match:
            start_idx = int(match.group(1))
            end_idx = int(match.group(2))
            indices = list(range(start_idx, end_idx + 1))
            name = focal_fg_alphacarbon
            return indices, name
        m_single = re.search(r"\d+", focal_fg_alphacarbon)
        if m_single:
            idx: int = int(m_single.group(0))
            return [idx], focal_fg_alphacarbon
        raise ValueError(f"Could not parse focal_fg_alphacarbon string: {focal_fg_alphacarbon}")

    idx = int(focal_fg_alphacarbon)
    return [idx], f"fgC{idx}"


def compute_native_states_and_profile(
    native_pdb_path: str,
    kap_n_ca: int,
    focal_fg_alphacarbon: int | tuple[int, int] | list[int] | str,
    interaction_capacity: int,
    max_surface_dist: float,
    unique_components: np.ndarray,
    cluster_centers: np.ndarray,
    similarity_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the native PDB interaction profile and match against Markov cluster centers."""
    pdb: md.Trajectory = md.load(native_pdb_path)
    all_ca: np.ndarray = pdb.topology.select("name CA")
    kap_ca: np.ndarray = all_ca[:kap_n_ca]

    focal_indices, focal_comp = extract_focal_info(focal_fg_alphacarbon=focal_fg_alphacarbon)
    fg_ca_indices: np.ndarray = np.array([all_ca[kap_n_ca + idx] for idx in focal_indices], dtype=int)

    kap_xyz: np.ndarray = np.transpose(pdb.xyz[:, kap_ca, :], (1, 2, 0))
    fg_xyz_raw: np.ndarray = np.transpose(pdb.xyz[:, fg_ca_indices, :], (1, 2, 0))
    fg_xyz: np.ndarray = np.mean(fg_xyz_raw, axis=0, keepdims=True)

    window_size: int = 100
    kap_fake: np.ndarray = np.repeat(kap_xyz, window_size, axis=2)
    fg_fake: np.ndarray = np.repeat(fg_xyz, window_size, axis=2)

    full_coords: np.ndarray = np.concatenate((kap_fake, fg_fake), axis=0)
    components: np.ndarray = np.array([f"kapC{i}" for i in range(kap_n_ca)] + [focal_comp])

    fake_input: iMSMInput = iMSMInput(
        trajectories=[iMSMInputSingleSim(trajectory=full_coords)],
        components=components,
        focal_component=focal_comp,
    )
    fake_config: iMSMConfig = iMSMConfig(
        checkpoints_path=".",
        window_size=window_size,
        interaction_capacity=interaction_capacity,
        max_surface_dist=max_surface_dist,
        n_cpus=1,
    )

    fake_cat = default_categorize(fake_input, fake_config)
    fake_emb = default_embed(fake_cat, fake_config)
    native_raw_hist: np.ndarray = fake_emb.trajectories[0].trajectory[0, :, 0]

    comp_to_val: dict[str, float] = {
        str(c): float(val) for c, val in zip(fake_emb.unique_components, native_raw_hist)
    }
    aligned_native_hist: np.ndarray = np.array(
        [comp_to_val.get(str(c), 0.0) for c in unique_components], dtype=float
    )

    norm_centers: np.ndarray = np.linalg.norm(cluster_centers, axis=1)
    norm_native: float = float(np.linalg.norm(aligned_native_hist))
    similarities: np.ndarray = np.dot(cluster_centers, aligned_native_hist) / (
        norm_centers * norm_native + 1e-12
    )
    native_state_indices: np.ndarray = np.where(similarities >= similarity_threshold)[0]

    return native_state_indices, similarities, aligned_native_hist


def compute_heat_repeats_alignment_matrix(
    kap_xyz: np.ndarray,
    heat5_range: tuple[int, int],
    heat6_range: tuple[int, int],
) -> tuple[np.ndarray, float, float, float]:
    """Compute rotation matrix and Euler angles aligning HEAT repeats 5 & 6 plane to camera view."""
    p5: np.ndarray = kap_xyz[heat5_range[0] : heat5_range[1] + 1]
    p6: np.ndarray = kap_xyz[heat6_range[0] : heat6_range[1] + 1]

    # Direction vectors along HEAT repeats
    v5: np.ndarray = p5[-1] - p5[0]
    v6: np.ndarray = p6[-1] - p6[0]
    if float(np.dot(v5, v6)) < 0.0:
        v6 = -v6

    # Average direction along repeats -> aligns with Y-axis
    v_along: np.ndarray = (v5 + v6) / 2.0
    y_axis: np.ndarray = v_along / (np.linalg.norm(v_along) + 1e-12)

    # Vector between HEAT repeat centers -> aligns with X-axis
    c5: np.ndarray = np.mean(p5, axis=0)
    c6: np.ndarray = np.mean(p6, axis=0)
    v_between: np.ndarray = c6 - c5

    # Orthogonalize X relative to Y
    v_between_perp: np.ndarray = v_between - float(np.dot(v_between, y_axis)) * y_axis
    x_axis: np.ndarray = v_between_perp / (np.linalg.norm(v_between_perp) + 1e-12)

    # Normal vector perpendicular to the HEAT repeats plane -> camera Z-axis
    z_axis: np.ndarray = np.cross(x_axis, y_axis)
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-12)

    # Ensure Z-axis points outward toward the binding surface (away from Kap95 bulk center)
    kap_center: np.ndarray = np.mean(kap_xyz, axis=0)
    heat_center: np.ndarray = (c5 + c6) / 2.0
    v_out: np.ndarray = heat_center - kap_center
    if float(np.dot(z_axis, v_out)) < 0.0:
        z_axis = -z_axis
        x_axis = -x_axis

    rot_mat: np.ndarray = np.stack([x_axis, y_axis, z_axis], axis=0)

    # Decompose into Euler angles for VMD (Ry -> Rx -> Rz)
    sin_x: float = float(np.clip(rot_mat[2, 1], -1.0, 1.0))
    theta_x: float = float(np.arcsin(sin_x))
    theta_y: float = float(np.arctan2(-rot_mat[2, 0], rot_mat[2, 2]))
    theta_z: float = float(np.arctan2(-rot_mat[0, 1], rot_mat[1, 1]))

    rot_x_deg: float = float(np.degrees(theta_x))
    rot_y_deg: float = float(np.degrees(theta_y))
    rot_z_deg: float = float(np.degrees(theta_z))

    return rot_mat, rot_x_deg, rot_y_deg, rot_z_deg


def reorder_imsm_states_by_y(
    imsm_checkpoint_path: str,
    top_path: str,
    kap_n_ca: int,
    heat5_range: tuple[int, int],
    heat6_range: tuple[int, int],
    tm_prior: float,
) -> np.ndarray:
    """Reorder Markov state IDs in clustering and MSM checkpoints by descending spatial Y coordinate."""
    with open(f"{imsm_checkpoint_path}/3_clustering.pickle", "rb") as f_cluster:
        imsm_cluster = pickle.load(f_cluster)

    cluster_centers: np.ndarray = np.array(imsm_cluster.clustering.cluster_centers_)
    comps: np.ndarray = np.array(imsm_cluster.unique_components)
    n_clusters: int = int(cluster_centers.shape[0])

    pdb: md.Trajectory = md.load(top_path)
    all_ca: np.ndarray = pdb.topology.select("name CA")
    kap_ca: np.ndarray = all_ca[:kap_n_ca]
    kap_xyz: np.ndarray = pdb.xyz[0, kap_ca, :]

    rot_mat, _, _, _ = compute_heat_repeats_alignment_matrix(
        kap_xyz=kap_xyz,
        heat5_range=heat5_range,
        heat6_range=heat6_range,
    )

    kap_weights: np.ndarray = np.zeros((n_clusters, kap_n_ca), dtype=float)
    for col_idx, comp_name in enumerate(comps):
        if str(comp_name).startswith("kapC"):
            res_idx: int = int(str(comp_name)[4:])
            if res_idx < kap_n_ca:
                kap_weights[:, res_idx] = cluster_centers[:, col_idx]

    sums: np.ndarray = kap_weights.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    norm_weights: np.ndarray = kap_weights / sums
    state_xyz: np.ndarray = np.dot(norm_weights, kap_xyz)

    kap_center: np.ndarray = np.mean(kap_xyz, axis=0, keepdims=True)
    state_xyz_centered: np.ndarray = state_xyz - kap_center
    state_xyz_rot: np.ndarray = np.dot(state_xyz_centered, rot_mat.T)
    y_coords: np.ndarray = state_xyz_rot[:, 1]

    # Order from highest Y to lowest Y
    order: np.ndarray = np.argsort(-y_coords)
    rank_map: np.ndarray = np.zeros(n_clusters, dtype=int)
    rank_map[order] = np.arange(n_clusters)

    # Reorder cluster centers
    imsm_cluster.clustering.cluster_centers_ = cluster_centers[order]
    if hasattr(imsm_cluster.clustering, "labels_") and imsm_cluster.clustering.labels_ is not None:
        imsm_cluster.clustering.labels_ = rank_map[imsm_cluster.clustering.labels_]

    # Remap discrete trajectory states
    for sim_clust in imsm_cluster.trajectories:
        sim_clust.trajectory = rank_map[sim_clust.trajectory]

    # Recompute MSM transition matrix from reordered trajectories
    config: iMSMConfig = iMSMConfig(checkpoints_path=imsm_checkpoint_path)
    config.tm_prior = tm_prior
    new_msm: iMSMMSM = default_msm(clustering=imsm_cluster, config=config)

    # Save updated checkpoints
    with open(f"{imsm_checkpoint_path}/3_clustering.pickle", "wb") as f_cluster_out:
        pickle.dump(imsm_cluster, f_cluster_out)

    with open(f"{imsm_checkpoint_path}/4_msm.pickle", "wb") as f_msm_out:
        pickle.dump(new_msm, f_msm_out)

    print(f"\n[Reordered States by Y (Descending)] Checkpoint: {imsm_checkpoint_path}")
    for new_id, old_id in enumerate(order):
        print(f"  State {new_id} <- Old State {old_id} (Rotated Y = {y_coords[old_id]:.3f} nm)")

    return order


def visualize_kap_states_and_rates(
    top_path: str,
    imsm_checkpoint_path: str,
    output_plot_path: str | None,
    top_rate_percentile: float,
    kap_n_ca: int,
    repeat_colors: tuple[str, str],
    zoom: bool,
    state_size_factor: float,
    free_threshold: float,
    dt_ns: float,
    auto_align_plane: bool,
    rotation_x_deg: float,
    rotation_y_deg: float,
    rotation_z_deg: float,
    heat5_range: tuple[int, int],
    heat6_range: tuple[int, int],
    focal_fg_alphacarbon: int | tuple[int, int] | list[int] | str,
    native_pdb_path: str,
    interaction_capacity: int,
    max_surface_dist: float,
    native_similarity_threshold: float,
    top_n_print: int,
    title: str | None,
    show_full_kap: bool,
    heat_ribbon_width: float,
    heat_ribbon_alpha: float,
    state_color_mode: str,
    arrow_color: str | None,
    show_rate_colorbar: bool,
    show_residue_colorbar: bool,
    diverging_palette: tuple[str, ...],
    residue_color_range: tuple[int, int],
) -> plt.Figure:
    """Visualize Markov state spatial centroids and transition rates projected onto the 2D plane of Kap95."""
    _, focal_label = extract_focal_info(focal_fg_alphacarbon=focal_fg_alphacarbon)
    pdb: md.Trajectory = md.load(top_path)
    all_ca: np.ndarray = pdb.topology.select("name CA")
    kap_ca: np.ndarray = all_ca[:kap_n_ca]
    kap_xyz: np.ndarray = pdb.xyz[0, kap_ca, :]  # shape: (kap_n_ca, 3)

    with open(f"{imsm_checkpoint_path}/3_clustering.pickle", "rb") as f_cluster:
        imsm_cluster = pickle.load(f_cluster)
    cluster_centers: np.ndarray = np.array(imsm_cluster.clustering.cluster_centers_)
    comps: np.ndarray = np.array(imsm_cluster.unique_components)

    with open(f"{imsm_checkpoint_path}/4_msm.pickle", "rb") as f_msm:
        imsm_msm = pickle.load(f_msm)
    msm_mat: np.ndarray = imsm_msm.transition_matrix

    n_clusters: int = msm_mat.shape[0]

    # Compute infinitesimal generator matrix Q (rates in µs⁻¹)
    dt_us: float = dt_ns / 1000.0
    generator_mat: np.ndarray = infinitesimal_generator(msm_mat, dt=dt_us)
    rate_mat: np.ndarray = generator_mat.copy()
    np.fill_diagonal(rate_mat, 0.0)
    rate_mat = np.maximum(rate_mat, 0.0)

    # 1. Identify Free State (cluster with highest unbound fraction, only if >= free_threshold)
    unbound_matches: np.ndarray = np.array(
        [i for i, c in enumerate(comps) if str(c).strip().lower() == "unbound"], dtype=int
    )
    unbound_col_idx: int = int(unbound_matches[0]) if len(unbound_matches) > 0 else -1
    unbound_weights: np.ndarray = (
        cluster_centers[:, unbound_col_idx] if unbound_col_idx >= 0 else np.zeros(n_clusters, dtype=float)
    )
    candidate_free_idx: int = int(np.argmax(unbound_weights)) if len(unbound_weights) > 0 else 0
    free_state_idx: int | None = (
        candidate_free_idx if (len(unbound_weights) > 0 and float(unbound_weights[candidate_free_idx]) >= free_threshold) else None
    )

    # 2. Identify Native State(s) from static PDB interaction profile
    native_state_indices: np.ndarray
    similarities: np.ndarray
    native_hist: np.ndarray
    native_state_indices, similarities, native_hist = compute_native_states_and_profile(
        native_pdb_path=native_pdb_path,
        kap_n_ca=kap_n_ca,
        focal_fg_alphacarbon=focal_fg_alphacarbon,
        interaction_capacity=interaction_capacity,
        max_surface_dist=max_surface_dist,
        unique_components=comps,
        cluster_centers=cluster_centers,
        similarity_threshold=native_similarity_threshold,
    )

    print("\n=======================================================")
    print(
        f"Markov State Analysis (Focal Bead {focal_label}, Native Similarity Thresh: {native_similarity_threshold:.2f}):"
    )
    interact_strs: list[str] = []
    for i in range(n_clusters):
        unbound_val: float = float(unbound_weights[i]) if (unbound_col_idx >= 0 and len(unbound_weights) > i) else 0.0
        inter_pct: float = max(0.0, min(1.0, 1.0 - unbound_val)) * 100.0
        free_str: str = " (Free)" if (free_state_idx is not None and i == free_state_idx) else ""
        interact_strs.append(f"State {i}: {inter_pct:.1f}%{free_str}")
    print(f"Total Interacting %: {', '.join(interact_strs)}")
    sim_strs: list[str] = [f"State {i}: {similarities[i]:.3f}" for i in range(n_clusters)]
    print(f"Cluster Cosine Similarities: {', '.join(sim_strs)}")
    print(f"Matched native-like states: {list(native_state_indices)}")
    print("=======================================================")
    print_state_makeup(
        title="Native PDB Conformation Composition",
        histogram=native_hist,
        components=comps,
        top_n=top_n_print,
    )
    for state_idx in native_state_indices:
        print_state_makeup(
            title=f"Matched Cluster State {state_idx} (Cosine Sim: {similarities[state_idx]:.3f}) Centroid Composition",
            histogram=cluster_centers[state_idx],
            components=comps,
            top_n=top_n_print,
        )

    # 3. Weighted average 3D spatial coordinates on Kap95 C-alphas (mapped by residue index)
    kap_weights: np.ndarray = np.zeros((n_clusters, kap_n_ca), dtype=float)
    for col_idx, comp_name in enumerate(comps):
        if str(comp_name).startswith("kapC"):
            res_idx: int = int(str(comp_name)[4:])
            if res_idx < kap_n_ca:
                kap_weights[:, res_idx] = cluster_centers[:, col_idx]

    sums: np.ndarray = kap_weights.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    norm_weights: np.ndarray = kap_weights / sums
    state_xyz: np.ndarray = np.dot(norm_weights, kap_xyz)  # shape: (n_clusters, 3)

    bound_mask: np.ndarray = (
        np.ones(n_clusters, dtype=bool)
        if free_state_idx is None
        else np.arange(n_clusters) != free_state_idx
    )
    bound_states_3d: np.ndarray = state_xyz[bound_mask]

    effective_rot_x: float = rotation_x_deg
    effective_rot_y: float = rotation_y_deg
    effective_rot_z: float = rotation_z_deg
    rot_matrix: np.ndarray

    if auto_align_plane:
        rot_matrix, effective_rot_x, effective_rot_y, effective_rot_z = (
            compute_heat_repeats_alignment_matrix(
                kap_xyz=kap_xyz,
                heat5_range=heat5_range,
                heat6_range=heat6_range,
            )
        )
        print("\n=======================================================")
        print("[Auto-Align Plane] Optimal camera viewing angles for HEAT repeats plane:")
        print(f"  rotation_x_deg = {effective_rot_x:.2f}°")
        print(f"  rotation_y_deg = {effective_rot_y:.2f}°")
        print(f"  rotation_z_deg = {effective_rot_z:.2f}°")
        print(f"  HEAT repeats plane normal: [{rot_matrix[2, 0]:.4f}, {rot_matrix[2, 1]:.4f}, {rot_matrix[2, 2]:.4f}]")
        print("=======================================================\n")
    else:
        theta_x_rad: float = float(np.radians(effective_rot_x))
        theta_y_rad: float = float(np.radians(effective_rot_y))
        theta_z_rad: float = float(np.radians(effective_rot_z))

        cos_tx: float = float(np.cos(theta_x_rad))
        sin_tx: float = float(np.sin(theta_x_rad))
        rot_x: np.ndarray = np.array(
            [[1.0, 0.0, 0.0], [0.0, cos_tx, -sin_tx], [0.0, sin_tx, cos_tx]], dtype=float
        )

        cos_ty: float = float(np.cos(theta_y_rad))
        sin_ty: float = float(np.sin(theta_y_rad))
        rot_y: np.ndarray = np.array(
            [[cos_ty, 0.0, sin_ty], [0.0, 1.0, 0.0], [-sin_ty, 0.0, cos_ty]], dtype=float
        )

        cos_tz: float = float(np.cos(theta_z_rad))
        sin_tz: float = float(np.sin(theta_z_rad))
        rot_z: np.ndarray = np.array(
            [[cos_tz, -sin_tz, 0.0], [sin_tz, cos_tz, 0.0], [0.0, 0.0, 1.0]], dtype=float
        )

        rot_matrix = np.dot(rot_z, np.dot(rot_x, rot_y))

    # 4. Rotate 3D coordinates around axes (centered on Kap95) before 2D projection
    kap_center: np.ndarray = np.mean(kap_xyz, axis=0, keepdims=True)
    kap_xyz_centered: np.ndarray = kap_xyz - kap_center
    state_xyz_centered: np.ndarray = state_xyz - kap_center

    kap_xyz_rot: np.ndarray = np.dot(kap_xyz_centered, rot_matrix.T)
    state_xyz_rot: np.ndarray = np.dot(state_xyz_centered, rot_matrix.T)

    kap_2d: np.ndarray = kap_xyz_rot[:, :2]
    state_2d: np.ndarray = state_xyz_rot[:, :2].copy()

    bound_mask = (
        np.ones(n_clusters, dtype=bool)
        if free_state_idx is None
        else np.arange(n_clusters) != free_state_idx
    )
    bound_states_2d: np.ndarray = state_2d[bound_mask]

    x_lim_min: float = 0.0
    x_lim_max: float = 0.0
    y_lim_min: float = 0.0
    y_lim_max: float = 0.0

    if zoom:
        x_min_bound: float = float(np.min(bound_states_2d[:, 0]))
        x_max_bound: float = float(np.max(bound_states_2d[:, 0]))
        y_min_bound: float = float(np.min(bound_states_2d[:, 1]))
        y_max_bound: float = float(np.max(bound_states_2d[:, 1]))

        center_x: float = float((x_min_bound + x_max_bound) / 2.0)
        center_y: float = float((y_min_bound + y_max_bound) / 2.0)

        # Shift all 2D coordinates so (0, 0) is at the center of the figure
        center_offset: np.ndarray = np.array([center_x, center_y])
        kap_2d = kap_2d - center_offset
        state_2d = state_2d - center_offset

        # Show a tightly zoomed area centered on bound states
        target_span: float = 1.2
        span_x: float = max(x_max_bound - x_min_bound + 0.5, target_span)
        span_y: float = max(y_max_bound - y_min_bound + 0.5, target_span)
        zoom_span: float = max(span_x, span_y)
        half_span: float = zoom_span / 2.0

        x_lim_min = -half_span
        x_lim_max = half_span
        y_lim_min = -half_span
        y_lim_max = half_span

        # Position Free State at top right corner relative to zoom box if present
        if free_state_idx is not None:
            state_2d[free_state_idx] = np.array([x_lim_max - zoom_span * 0.15, y_lim_max - zoom_span * 0.15])
    else:
        x_min: float = float(np.min(kap_2d[:, 0]))
        x_max: float = float(np.max(kap_2d[:, 0]))
        y_min: float = float(np.min(kap_2d[:, 1]))
        y_max: float = float(np.max(kap_2d[:, 1]))

        center_x = float((x_min + x_max) / 2.0)
        center_y = float((y_min + y_max) / 2.0)

        center_offset = np.array([center_x, center_y])
        kap_2d = kap_2d - center_offset
        state_2d = state_2d - center_offset

        # Position Free State at top right corner of bounding box if present
        if free_state_idx is not None:
            x_max_shifted: float = float(np.max(kap_2d[:, 0]))
            y_max_shifted: float = float(np.max(kap_2d[:, 1]))
            state_2d[free_state_idx] = np.array([x_max_shifted * 0.9, y_max_shifted * 1.1])

    # 5. Stationary distribution calculation
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    eigenvalues, eigenvectors = np.linalg.eig(msm_mat.T)
    idx: int = int(np.argmin(np.abs(eigenvalues - 1.0)))
    stat_dist: np.ndarray = np.real(eigenvectors[:, idx])
    stat_dist = stat_dist / np.sum(stat_dist)

    fig, ax = plt.subplots(figsize=(13.0, 9.5))

    # Fit smooth cubic spline over Kap95 2D C-alpha backbone
    res_indices: np.ndarray = np.arange(kap_n_ca, dtype=float)
    spline_2d: CubicSpline = CubicSpline(res_indices, kap_2d, bc_type="natural")

    # Kap95 HEAT repeat residue boundaries (19 repeats across 861 residues)
    repeat_boundaries: list[int] = [
        0, 36, 89, 133, 176, 218, 259, 316, 366, 401, 451, 495, 535, 591, 633, 674, 717, 772, 818, kap_n_ca,
    ]

    base_ribbon_w: float = 4.0 if zoom else 3.5
    kap_ribbon_alpha: float = 0.05 if zoom else 0.25
    kap_border_alpha: float = 0.03 if zoom else 0.15
    kap_shine_alpha: float = 0.03 if zoom else 0.20

    # Smooth 3D ribbon backbone trace for other HEAT repeats
    if show_full_kap:
        for r_idx in range(len(repeat_boundaries) - 1):
            start_r: int = repeat_boundaries[r_idx]
            end_r: int = repeat_boundaries[r_idx + 1]
            c: str = repeat_colors[r_idx % 2]

            n_interp: int = max(int((end_r - start_r) * 15), 30)
            t_seg: np.ndarray = np.linspace(start_r, min(end_r, kap_n_ca - 1), n_interp)
            seg_coords: np.ndarray = spline_2d(t_seg)

            label_name: str | None = (
                None
                if zoom
                else ("Kap95 Repeat (Even)" if r_idx == 0 else ("Kap95 Repeat (Odd)" if r_idx == 1 else None))
            )

            # 1. Dark outer border / contour for 3D depth separation
            ax.plot(
                seg_coords[:, 0],
                seg_coords[:, 1],
                color="#263238",
                alpha=kap_border_alpha,
                linewidth=base_ribbon_w + 1.6,
                solid_capstyle="round",
                solid_joinstyle="round",
                zorder=2,
            )
            # 2. Main colored ribbon body
            ax.plot(
                seg_coords[:, 0],
                seg_coords[:, 1],
                color=c,
                alpha=kap_ribbon_alpha,
                linewidth=base_ribbon_w,
                solid_capstyle="round",
                solid_joinstyle="round",
                zorder=2,
                label=label_name,
            )
            # 3. Specular central highlight for 3D cylindrical sheen
            ax.plot(
                seg_coords[:, 0],
                seg_coords[:, 1],
                color="white",
                alpha=kap_shine_alpha,
                linewidth=base_ribbon_w * 0.35,
                solid_capstyle="round",
                solid_joinstyle="round",
                zorder=3,
            )

    # Highlight specific HEAT repeat regions (single solid color each, slightly transparent)
    dark_red_color: str = "#8B0000"
    light_red_color: str = "#FF6B6B"
    highlight_ribbon_w: float = heat_ribbon_width
    highlight_ribbon_alpha: float = heat_ribbon_alpha

    # HEAT 5
    seg1_start: int = heat5_range[0]
    seg1_end: int = heat5_range[1]
    t_seg1: np.ndarray = np.linspace(seg1_start, min(seg1_end, kap_n_ca - 1), (seg1_end - seg1_start) * 20)
    seg1_coords: np.ndarray = spline_2d(t_seg1)

    ax.plot(
        seg1_coords[:, 0],
        seg1_coords[:, 1],
        color=dark_red_color,
        alpha=highlight_ribbon_alpha,
        linewidth=highlight_ribbon_w,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=3,
    )

    # HEAT 6
    seg2_start: int = heat6_range[0]
    seg2_end: int = heat6_range[1]
    t_seg2: np.ndarray = np.linspace(seg2_start, min(seg2_end, kap_n_ca - 1), (seg2_end - seg2_start) * 20)
    seg2_coords: np.ndarray = spline_2d(t_seg2)

    ax.plot(
        seg2_coords[:, 0],
        seg2_coords[:, 1],
        color=light_red_color,
        alpha=highlight_ribbon_alpha,
        linewidth=highlight_ribbon_w,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=3,
    )

    # Rate colormap: slowest (light gray/white) -> fastest (black)
    rate_cmap: LinearSegmentedColormap = LinearSegmentedColormap.from_list(
        "rate_bw", ["#D6D6D6", "#000000"]
    )

    # Precalculate Kap C-alpha bins for state_color_mode == "residue_index" (or "y_position")
    h5_center: float = (heat5_range[0] + heat5_range[1]) / 2.0
    h6_center: float = (heat6_range[0] + heat6_range[1]) / 2.0
    split_threshold: float = (h5_center + h6_center) / 2.0

    residue_indices: np.ndarray = np.arange(kap_n_ca, dtype=float)
    residue_offsets: np.ndarray = np.where(
        residue_indices < split_threshold,
        residue_indices - heat5_range[0] + 1.0,
        residue_indices - heat6_range[0] + 1.0,
    )
    min_offset: float = float(residue_color_range[0])
    max_offset: float = float(residue_color_range[1])
    n_palette_bins: int = len(diverging_palette)
    offset_bin_edges: np.ndarray = np.linspace(min_offset, max_offset, n_palette_bins + 1)
    kap_bin_indices: np.ndarray = np.clip(
        np.digitize(residue_offsets, offset_bin_edges[1:-1]),
        0,
        n_palette_bins - 1,
    )

    # State nodes sizing
    node_sizes: np.ndarray = (250.0 + 1400.0 * (stat_dist / np.max(stat_dist))) * state_size_factor
    node_radii: np.ndarray = np.sqrt(node_sizes) / 2.0

    positive_rates: np.ndarray = rate_mat[rate_mat > 0]
    top_threshold: float = (
        float(np.percentile(positive_rates, top_rate_percentile)) if len(positive_rates) > 0 else 0.0
    )
    max_rate: float = float(np.max(rate_mat)) if np.max(rate_mat) > 0 else 1.0

    for i in range(n_clusters):
        for j in range(n_clusters):
            if i == j:
                continue
            rate: float = float(rate_mat[i, j])
            if rate >= top_threshold and rate > 0:
                norm_rate: float = rate / max_rate
                r_a: float = float(node_radii[i]) * 1.15
                r_b: float = float(node_radii[j]) * 1.15
                arrow_c = "black" if arrow_color == "black" else rate_cmap(norm_rate)
                arrow: FancyArrowPatch = FancyArrowPatch(
                    posA=(state_2d[i, 0], state_2d[i, 1]),
                    posB=(state_2d[j, 0], state_2d[j, 1]),
                    connectionstyle="arc3,rad=0.18",
                    arrowstyle="-|>",
                    mutation_scale=16,
                    shrinkA=r_a,
                    shrinkB=r_b,
                    linewidth=1.0 + 3.0 * norm_rate,
                    color=arrow_c,
                    alpha=0.85,
                    zorder=4,
                )
                ax.add_patch(arrow)

    # Render each state node with exact original scatter sizing
    for i in range(n_clusters):
        unbound_frac: float = float(unbound_weights[i]) if (unbound_col_idx >= 0 and len(unbound_weights) > i) else 0.0
        interact_frac: float = max(0.0, min(1.0, 1.0 - unbound_frac))
        is_free: bool = (free_state_idx is not None) and (i == free_state_idx)

        # 1. Uninteracting background circle (empty / light region)
        ax.scatter(
            state_2d[i, 0],
            state_2d[i, 1],
            s=node_sizes[i],
            marker="o",
            facecolors="#F5F7FA",
            edgecolors="none",
            zorder=5.0,
        )

        # 2. Interacting pie slices
        if interact_frac > 0.001:
            if state_color_mode in ("residue_index", "y_position"):
                total_state_kap_w: float = float(np.sum(kap_weights[i, :]))
                if total_state_kap_w > 0.0:
                    bin_weights: np.ndarray = np.zeros(n_palette_bins, dtype=float)
                    for b_idx in range(n_palette_bins):
                        bin_weights[b_idx] = float(np.sum(kap_weights[i, kap_bin_indices == b_idx]))
                    bin_fractions: np.ndarray = bin_weights / total_state_kap_w

                    theta_curr: float = 90.0
                    for b_idx in range(n_palette_bins):
                        slice_angle: float = float(bin_fractions[b_idx] * interact_frac * 360.0)
                        if slice_angle > 0.05:
                            theta_next: float = theta_curr + slice_angle
                            wedge_marker: mpath.Path = create_wedge_marker(theta1_deg=theta_curr, theta2_deg=theta_next)
                            ax.scatter(
                                state_2d[i, 0],
                                state_2d[i, 1],
                                s=node_sizes[i],
                                marker=wedge_marker,
                                facecolors=diverging_palette[b_idx],
                                edgecolors="none",
                                zorder=5.1,
                            )
                            theta_curr = theta_next
            else:
                theta_end: float = 90.0 + interact_frac * 360.0
                wedge_marker = create_wedge_marker(theta1_deg=90.0, theta2_deg=theta_end)
                ax.scatter(
                    state_2d[i, 0],
                    state_2d[i, 1],
                    s=node_sizes[i],
                    marker=wedge_marker,
                    facecolors="#4A6572",
                    edgecolors="none",
                    zorder=5.1,
                )

        # 3. Full circular outer border (always a complete circle)
        ax.scatter(
            state_2d[i, 0],
            state_2d[i, 1],
            s=node_sizes[i],
            marker="o",
            facecolors="none",
            edgecolors="#E53935" if is_free else "black",
            linewidths=2.2 if is_free else 1.5,
            zorder=5.2,
        )

        # 4. Accent ring for native-like states
        if i in native_state_indices:
            ax.scatter(
                state_2d[i, 0],
                state_2d[i, 1],
                s=node_sizes[i] * 1.55,
                facecolors="none",
                edgecolors="#FF8C00",
                linewidths=3.0,
                linestyle="-",
                zorder=5.3,
            )

        label_str: str = f"{i} (Free)" if is_free else str(i)
        base_fontsize: float = 12.0 if is_free else 13.5
        label_fontsize: float = base_fontsize * float(np.sqrt(state_size_factor) / 2.0)
        ax.annotate(
            label_str,
            (state_2d[i, 0], state_2d[i, 1]),
            ha="center",
            va="center",
            fontsize=label_fontsize,
            fontweight="bold",
            color="#E53935" if is_free else "#0D1B2A",
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
            zorder=6,
        )

    if show_rate_colorbar:
        sm_rate = plt.cm.ScalarMappable(cmap=rate_cmap, norm=plt.Normalize(vmin=0.0, vmax=max_rate))
        sm_rate.set_array([])
        cbar_rate = fig.colorbar(sm_rate, ax=ax, shrink=0.75, pad=0.02)
        cbar_rate.set_label("Transition rate (µs⁻¹)", fontsize=25.5)
        cbar_rate.ax.tick_params(labelsize=23.25)

    if show_residue_colorbar and len(diverging_palette) > 0:
        diverging_cmap: LinearSegmentedColormap = LinearSegmentedColormap.from_list(
            "kap_residue_palette", list(diverging_palette), N=256
        )
        sm_res = plt.cm.ScalarMappable(
            cmap=diverging_cmap,
            norm=plt.Normalize(vmin=min_offset, vmax=max_offset),
        )
        sm_res.set_array([])
        cbar_res = fig.colorbar(sm_res, ax=ax, shrink=0.75, pad=0.08)
        tick_vals: np.ndarray = np.linspace(min_offset, max_offset, 5)

        cbar_res.set_ticks(tick_vals)
        cbar_res.ax.set_yticklabels([str(int(round(val))) for val in tick_vals])
        cbar_res.ax.tick_params(labelsize=23.25)
        cbar_res.ax.set_title("Position in\n$\\alpha$-helix", fontsize=25.5, pad=14, fontweight="normal")

    title_suffix: str = " (Zoomed)" if zoom else ""
    plot_title: str = (
        title
        if title is not None
        else f"iMSM State Network & {focal_label} Transition Rates on Kap95 Structure{title_suffix}"
    )
    ax.set_title(
        plot_title,
        fontsize=20.0,
        fontweight="bold",
    )
    ax.set_xlabel(r"$x$ (nm)", fontsize=25.5)
    ax.set_ylabel(r"$y$ (nm)", fontsize=25.5)
    ax.tick_params(labelsize=23.25)

    heat_proxies: list[object] = [
        Line2D(
            [0],
            [0],
            color=dark_red_color,
            linewidth=3.5,
            alpha=0.75,
            label=f"HEAT 5 ({heat5_range[0]}–{heat5_range[1]})",
        ),
        Line2D(
            [0],
            [0],
            color=light_red_color,
            linewidth=3.5,
            alpha=0.75,
            label=f"HEAT 6 ({heat6_range[0]}–{heat6_range[1]})",
        ),
    ]
    repeat_proxies: list[object] = []
    if show_full_kap and not zoom:
        repeat_proxies = [
            Line2D([0], [0], color=repeat_colors[0], linewidth=3.5, label="Kap95 Repeat (Even)"),
            Line2D([0], [0], color=repeat_colors[1], linewidth=3.5, label="Kap95 Repeat (Odd)"),
        ]

    state_frac_proxies: list[object] = []
    if state_color_mode not in ("residue_index", "y_position"):
        state_frac_proxies.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Interacting fraction",
                markerfacecolor="#4A6572",
                markeredgecolor="#1A252C",
                markersize=18,
            )
        )
    state_frac_proxies.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Non-interacting fraction",
            markerfacecolor="#F5F7FA",
            markeredgecolor="#1A252C",
            markersize=18,
        )
    )

    custom_proxies: list[object] = repeat_proxies + heat_proxies + state_frac_proxies
    if len(native_state_indices) > 0:
        native_label_str: str = f"Native-like ({', '.join(map(str, native_state_indices))})"
        custom_proxies.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=native_label_str,
                markerfacecolor="none",
                markeredgecolor="#FF8C00",
                markeredgewidth=2.5,
                markersize=21,
            )
        )
    ax.legend(handles=custom_proxies, loc="upper left", fontsize=19.6)

    if zoom:
        ax.set_xlim(x_lim_min, x_lim_max)
        ax.set_ylim(y_lim_min, y_lim_max)

    ax.set_aspect("equal")

    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, symmetric=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, symmetric=True))

    plt.tight_layout()
    if output_plot_path is not None:
        os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
        fig.savefig(output_plot_path, bbox_inches="tight", dpi=300)
    plt.show()

    return fig


def plot_fg_spatial_network(
    top_path: str,
    imsm_checkpoint_path: str,
    output_plot_path: str | None,
    top_rate_percentile: float,
    kap_n_ca: int,
    repeat_colors: tuple[str, str],
    zoom: bool,
    state_size_factor: float,
    free_threshold: float,
    dt_ns: float,
    auto_align_plane: bool,
    rotation_x_deg: float,
    rotation_y_deg: float,
    rotation_z_deg: float,
    heat5_range: tuple[int, int],
    heat6_range: tuple[int, int],
    focal_fg_alphacarbon: int | tuple[int, int] | list[int] | str,
    native_pdb_path: str,
    interaction_capacity: int,
    max_surface_dist: float,
    native_similarity_threshold: float,
    top_n_print: int,
    title: str | None,
    show_full_kap: bool,
    heat_ribbon_width: float,
    heat_ribbon_alpha: float,
    state_color_mode: str,
    arrow_color: str | None,
    show_rate_colorbar: bool,
    show_residue_colorbar: bool,
    diverging_palette: tuple[str, ...],
    residue_color_range: tuple[int, int],
) -> plt.Figure:
    """Plot static spatial network visualization of Markov states and rates."""
    return visualize_kap_states_and_rates(
        top_path=top_path,
        imsm_checkpoint_path=imsm_checkpoint_path,
        output_plot_path=output_plot_path,
        top_rate_percentile=top_rate_percentile,
        kap_n_ca=kap_n_ca,
        repeat_colors=repeat_colors,
        zoom=zoom,
        state_size_factor=state_size_factor,
        free_threshold=free_threshold,
        dt_ns=dt_ns,
        auto_align_plane=auto_align_plane,
        rotation_x_deg=rotation_x_deg,
        rotation_y_deg=rotation_y_deg,
        rotation_z_deg=rotation_z_deg,
        heat5_range=heat5_range,
        heat6_range=heat6_range,
        focal_fg_alphacarbon=focal_fg_alphacarbon,
        native_pdb_path=native_pdb_path,
        interaction_capacity=interaction_capacity,
        max_surface_dist=max_surface_dist,
        native_similarity_threshold=native_similarity_threshold,
        top_n_print=top_n_print,
        title=title,
        show_full_kap=show_full_kap,
        heat_ribbon_width=heat_ribbon_width,
        heat_ribbon_alpha=heat_ribbon_alpha,
        state_color_mode=state_color_mode,
        arrow_color=arrow_color,
        show_rate_colorbar=show_rate_colorbar,
        show_residue_colorbar=show_residue_colorbar,
        diverging_palette=diverging_palette,
        residue_color_range=residue_color_range,
    )


def find_state_exemplar_frames(
    imsm_checkpoint_path: str,
    window_size: int,
    stride: int,
    first_frame: int | None,
    ns_per_frame: float,
) -> list[dict[str, int | float]]:
    """Find the most representative simulation frame for each Markov state based on minimum centroid distance."""
    with open(f"{imsm_checkpoint_path}/3_clustering.pickle", "rb") as f_cluster:
        imsm_cluster = pickle.load(f_cluster)
    cluster_centers: np.ndarray = np.array(imsm_cluster.clustering.cluster_centers_)
    n_clusters: int = cluster_centers.shape[0]

    with open(f"{imsm_checkpoint_path}/2_embed.pickle", "rb") as f_embed:
        imsm_embed = pickle.load(f_embed)
    embed_vectors: np.ndarray = imsm_embed.trajectories[0].trajectory[:, :, 0]

    exemplars: list[dict[str, int | float]] = []
    base_frame_offset: int = 0 if first_frame is None else first_frame

    for state_id in range(n_clusters):
        centroid: np.ndarray = cluster_centers[state_id]
        dists: np.ndarray = np.linalg.norm(embed_vectors - centroid, axis=1)
        best_window_idx: int = int(np.argmin(dists))
        best_dist: float = float(dists[best_window_idx])

        start_frame: int = base_frame_offset + (best_window_idx * stride)
        mid_frame: int = start_frame + (window_size // 2)
        sim_time_ns: float = mid_frame * ns_per_frame
        sim_time_us: float = sim_time_ns / 1000.0

        exemplars.append(
            {
                "state_id": state_id,
                "window_idx": best_window_idx,
                "mid_frame": mid_frame,
                "sim_time_ns": sim_time_ns,
                "sim_time_us": sim_time_us,
                "centroid_distance": best_dist,
            }
        )

    return exemplars


def generate_interaction_colorbar(
    output_path: str,
    low_color: str,
    high_color: str,
    figsize: tuple[float, float],
) -> None:
    """Generate and save a standalone horizontal log-scale colorbar for Kap95 interaction frequency."""
    custom_cmap: LinearSegmentedColormap = LinearSegmentedColormap.from_list(
        "kap_interaction_log", [low_color, high_color], N=1024
    )
    norm: LogNorm = LogNorm(vmin=0.001, vmax=1.0)

    fig, ax = plt.subplots(figsize=figsize)
    cb = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=custom_cmap),
        cax=ax,
        orientation="horizontal",
        ticks=[0.001, 0.01, 0.1, 1.0],
    )
    cb.ax.tick_params(labelsize=13.0)
    cb.ax.set_xticklabels([r"$\leq 0.1\%$", "1%", "10%", "100%"], fontsize=13.0, fontweight="normal")
    cb.set_label("Kap95 Interaction Frequency", fontsize=14.0, fontweight="normal", labelpad=6)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def generate_vmd_state_scripts(
    top_path: str,
    traj_paths: list[str],
    imsm_checkpoint_path: str,
    output_dir: str,
    kap_n_ca: int,
    focal_fg_alphacarbon: int | tuple[int, int] | list[int] | str,
    window_size: int,
    stride: int,
    first_frame: int | None,
    ns_per_frame: float,
    free_threshold: float,
    auto_align_plane: bool,
    rotation_x_deg: float,
    rotation_y_deg: float,
    rotation_z_deg: float,
    heat5_range: tuple[int, int],
    heat6_range: tuple[int, int],
    vmd_zoom_scale: float,
    all_states_translate: tuple[float, float, float],
) -> list[dict[str, int | float]]:
    """Generate VMD Tcl scripts and interactive procedures to view and capture images of each Markov state."""
    exemplars: list[dict[str, int | float]] = find_state_exemplar_frames(
        imsm_checkpoint_path=imsm_checkpoint_path,
        window_size=window_size,
        stride=stride,
        first_frame=first_frame,
        ns_per_frame=ns_per_frame,
    )

    pdb: md.Trajectory = md.load(top_path)
    all_ca: np.ndarray = pdb.topology.select("name CA")
    kap_ca: np.ndarray = all_ca[:kap_n_ca]
    fg1_ca: np.ndarray = all_ca[kap_n_ca : kap_n_ca + 125]

    focal_indices, focal_label = extract_focal_info(focal_fg_alphacarbon=focal_fg_alphacarbon)
    focal_ca_indices: list[int] = [int(fg1_ca[idx]) for idx in focal_indices]
    focal_atoms = [pdb.topology.atom(idx) for idx in focal_ca_indices]
    focal_res_seqs: list[int] = [int(atom.residue.resSeq) for atom in focal_atoms]
    focal_res_sel_str: str = " or ".join([f"resid {s}" for s in sorted(set(focal_res_seqs))])
    focal_names_str: str = "+".join([f"{atom.residue.name}{atom.residue.resSeq}" for atom in focal_atoms])

    kap_xyz_angstrom: np.ndarray = pdb.xyz[0, kap_ca, :] * 10.0

    # Compute contact cluster center for each state
    with open(f"{imsm_checkpoint_path}/3_clustering.pickle", "rb") as f_cluster:
        imsm_cluster = pickle.load(f_cluster)
    cluster_centers: np.ndarray = np.array(imsm_cluster.clustering.cluster_centers_)
    comps: np.ndarray = np.array(imsm_cluster.unique_components)

    unbound_matches: np.ndarray = np.where(comps == "Unbound")[0]
    unbound_col_idx: int = int(unbound_matches[0]) if len(unbound_matches) > 0 else -1
    unbound_weights: np.ndarray = cluster_centers[:, unbound_col_idx]
    candidate_free_idx: int = int(np.argmax(unbound_weights))
    free_state_idx: int | None = (
        candidate_free_idx if float(unbound_weights[candidate_free_idx]) >= free_threshold else None
    )

    kap_weights: np.ndarray = np.zeros((len(exemplars), kap_n_ca), dtype=float)
    for col_idx, comp_name in enumerate(comps):
        if str(comp_name).startswith("kapC"):
            res_idx: int = int(str(comp_name)[4:])
            if res_idx < kap_n_ca:
                kap_weights[:, res_idx] = cluster_centers[:, col_idx]

    sums: np.ndarray = kap_weights.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    norm_weights: np.ndarray = kap_weights / sums
    state_xyz_angstrom: np.ndarray = np.dot(norm_weights, kap_xyz_angstrom)

    # Convert interaction weights to log scale [0.001, 1.0] -> normalized [0.0, 1.0] Beta field for VMD
    log_weights: np.ndarray = np.zeros_like(kap_weights)
    valid_mask: np.ndarray = kap_weights >= 0.001
    log_weights[valid_mask] = np.clip((np.log10(kap_weights[valid_mask]) + 3.0) / 3.0, 0.0, 1.0)

    effective_rot_x: float = rotation_x_deg
    effective_rot_y: float = rotation_y_deg
    effective_rot_z: float = rotation_z_deg

    if auto_align_plane:
        _, effective_rot_x, effective_rot_y, effective_rot_z = (
            compute_heat_repeats_alignment_matrix(
                kap_xyz=kap_xyz_angstrom / 10.0,
                heat5_range=heat5_range,
                heat6_range=heat6_range,
            )
        )

    os.makedirs(output_dir, exist_ok=True)
    snapshots_dir: str = os.path.abspath(os.path.join(output_dir, "snapshots"))
    os.makedirs(snapshots_dir, exist_ok=True)

    # Generate and display single shared interaction colorbar across all states (log scale 0.001 to 1.0)
    colorbar_path: str = os.path.join(snapshots_dir, "interaction_colorbar.png")
    generate_interaction_colorbar(
        output_path=colorbar_path,
        low_color="#78909C",
        high_color="#D32F2F",
        figsize=(6.5, 0.75),
    )

    # Color IDs palette for distinguishing states in all-states overlay (strictly non-red to preserve HEAT 5/6 red contrast)
    state_color_ids: list[int] = [3, 0, 7, 11, 10, 12, 14, 13, 15, 5]

    # Precompute 1024-step color scale in Python and hardcode into VMD script (Slate Gray #78909C -> Red #D32F2F)
    low_rgb: tuple[float, float, float] = (0.470, 0.565, 0.612)
    high_rgb: tuple[float, float, float] = (0.827, 0.184, 0.184)
    n_vmd_colors: int = 1024
    color_scale_lines: list[str] = [
        "proc apply_custom_colorscale {} {",
        "    color scale midpoint 0.5",
        "    color scale min 0.0",
        "    color scale max 1.0",
    ]
    for i in range(n_vmd_colors):
        t_val: float = i / float(n_vmd_colors - 1)
        r_val: float = low_rgb[0] + t_val * (high_rgb[0] - low_rgb[0])
        g_val: float = low_rgb[1] + t_val * (high_rgb[1] - low_rgb[1])
        b_val: float = low_rgb[2] + t_val * (high_rgb[2] - low_rgb[2])
        color_scale_lines.append(f"    color change rgb {33 + i} {r_val:.4f} {g_val:.4f} {b_val:.4f}")
    color_scale_lines.append("}")

    # Generate master Tcl script
    master_script_lines: list[str] = [
        "# ==============================================================================",
        "# VMD Visualization Script for iMSM Markov States",
        f"# Focal Component: {focal_label} ({focal_names_str})",
        "# ==============================================================================",
        "",
        "# Per-state Kap95 residue log-scaled interaction weights (0.0 = <=0.001, 1.0 = 1.0)",
    ]

    for ex in exemplars:
        s_id: int = int(ex["state_id"])
        state_weights_str: str = " ".join(f"{float(w):.4f}" for w in log_weights[s_id, :])
        master_script_lines.append(f"set state_{s_id}_kap_weights [list {state_weights_str}]")

    master_script_lines.extend(
        [
            "",
            "proc apply_state_kap_weights {weights_list} {",
            '    set sel_ref [atomselect top "protein and resid 1 to 861" frame 0]',
            "    set res_list [$sel_ref get resid]",
            "    set beta_vals [list]",
            "    set n_weights [llength $weights_list]",
            "    foreach r $res_list {",
            "        set idx [expr {$r - 1}]",
            "        if {$idx >= 0 && $idx < $n_weights} {",
            "            lappend beta_vals [lindex $weights_list $idx]",
            "        } else {",
            "            lappend beta_vals 0.0",
            "        }",
            "    }",
            "    $sel_ref delete",
            "",
            "    set num_frames [molinfo top get numframes]",
            "    for {set f 0} {$f < $num_frames} {incr f} {",
            '        set sel_f [atomselect top "protein and resid 1 to 861" frame $f]',
            "        $sel_f set beta $beta_vals",
            "        $sel_f delete",
            "    }",
            "}",
            "",
        ]
    )

    master_script_lines.extend(color_scale_lines)

    master_script_lines.extend(
        [
            "",
            "proc setup_msm_representations {} {",
            "    # Remove existing representations",
            "    set nreps [molinfo top get numreps]",
            "    for {set i [expr {$nreps - 1}]} {$i >= 0} {incr i -1} {",
            "        mol delrep $i top",
            "    }",
            "",
            "    # Define color palette matching 2D network plot",
            "    color change rgb 7 0.078 0.612 0.149   ;# Focal FSFG Motif: Green (#149C26)",
            "    color change rgb 2 0.750 0.750 0.750   ;# FG Chain 2: Light Gray",
            "    color change rgb 8 1.000 1.000 1.000   ;# Background: Pure White",
            "",
            "    # Tune material transparency for background Kap95 ribbon (more transparent)",
            "    material change opacity Transparent 0.25",
            "",
            f"    # 1. Background Kap95 Non-HEAT Ribbon - Semi-Transparent Colored by Interaction Frequency",
            "    mol representation NewCartoon 0.28 10.0 4.1 0",
            "    mol color Beta",
            f'    mol selection "protein and resid 1 to 861 and not (resid {heat5_range[0]} to {heat5_range[1]} or resid {heat6_range[0]} to {heat6_range[1]})"',
            "    mol material Transparent",
            "    mol addrep top",
            "    set kap_rep_idx [expr {[molinfo top get numreps] - 1}]",
            "    mol scaleminmax top $kap_rep_idx 0.0 1.0",
            "",
            f"    # 2. HEAT Repeats 5 & 6 (resid {heat5_range[0]} to {heat5_range[1]} & {heat6_range[0]} to {heat6_range[1]}) - OPAQUE Solid Ribbon Colored by Interaction Frequency",
            "    mol representation NewCartoon 0.45 10.0 4.1 0",
            "    mol color Beta",
            f'    mol selection "protein and (resid {heat5_range[0]} to {heat5_range[1]} or resid {heat6_range[0]} to {heat6_range[1]})"',
            "    mol material AOChalky",
            "    mol addrep top",
            "    set heat_rep_idx [expr {[molinfo top get numreps] - 1}]",
            "    mol scaleminmax top $heat_rep_idx 0.0 1.0",
            "",
            f"    # 3. Focal FSFG 4 AAs ({focal_names_str}) Backbone - Solid Green Tube",
            "    mol representation Tube 0.35 16.0",
            "    mol color ColorID 7",
            f'    mol selection "({focal_res_sel_str})"',
            "    mol material AOChalky",
            "    mol addrep top",
            "",
            f"    # 4. Focal FSFG 4 AAs ({focal_names_str}) Side Chains (No Hydrogen) - Green Licorice",
            "    mol representation Licorice 0.28 12.0 12.0",
            "    mol color ColorID 7",
            f'    mol selection "({focal_res_sel_str}) and not hydrogen and (sidechain or name CA)"',
            "    mol material Glossy",
            "    mol addrep top",
            "",
            f"    # 5. Rest of FG Chain 1 - Transparent Smooth Tube",
            "    mol representation Tube 0.22 16.0",
            "    mol color ColorID 7",
            f'    mol selection "resid 862 to 986 and not ({focal_res_sel_str})"',
            "    mol material Transparent",
            "    mol addrep top",
            "",
            "    # 6. FG Chain 2 (resid 987 to 1111) - Subtle Transparent Smooth Tube",
            "    mol representation Tube 0.20 16.0",
            "    mol color ColorID 2",
            '    mol selection "resid 987 to 1111"',
            "    mol material Transparent",
            "    mol addrep top",
            "",
            "    # Apply custom color scale after all representations are created so VMD does not overwrite it",
            "    apply_custom_colorscale",
            "",
            "    # Display settings matching 2D plot projection",
            "    display projection Orthographic",
            "    display depthcue off",
            "    display backgroundgradient off",
            "    color Display Background white",
            "    axes location off",
            "}",
            "",
            "set exemplars_aligned 0",
            "proc align_all_exemplar_frames {} {",
            "    if {$::exemplars_aligned} {",
            "        return",
            "    }",
            f'    set sel_ref [atomselect top "protein and (resid {heat5_range[0]} to {heat5_range[1]} or resid {heat6_range[0]} to {heat6_range[1]}) and name CA" frame 0]',
            f"    set exemplar_frames [list {' '.join(str(e['mid_frame']) for e in exemplars)}]",
            "    foreach f $exemplar_frames {",
            "        if {$f != 0} {",
            f'            set sel_cur [atomselect top "protein and (resid {heat5_range[0]} to {heat5_range[1]} or resid {heat6_range[0]} to {heat6_range[1]}) and name CA" frame $f]',
            "            set trans_mat [measure fit $sel_cur $sel_ref]",
            '            set move_all [atomselect top "all" frame $f]',
            "            $move_all move $trans_mat",
            "            $sel_cur delete",
            "            $move_all delete",
            "        }",
            "    }",
            "    # Translate frame 0 and all exemplar frames so the HEAT 5/6 midpoint is exactly at (0, 0, 0)",
            "    set heat_center [measure center $sel_ref]",
            "    set shift [vecscale -1.0 $heat_center]",
            "    set all_frames_to_shift [lsort -unique [concat $exemplar_frames 0]]",
            "    foreach f $all_frames_to_shift {",
            '        set move_all [atomselect top "all" frame $f]',
            "        $move_all moveby $shift",
            "        $move_all delete",
            "    }",
            "    $sel_ref delete",
            "    set ::exemplars_aligned 1",
            "}",
            "",
            "proc align_view_to_state {rot_x rot_y rot_z zoom_scale} {",
            "    align_all_exemplar_frames",
            "    display resetview",
            "    translate to 0.0 0.0 0.0",
            "    molinfo top set center [list {0.0 0.0 0.0}]",
            "    rotate y by $rot_y",
            "    rotate x by $rot_x",
            "    rotate z by $rot_z",
            "    scale by $zoom_scale",
            f"    translate by {all_states_translate[0]:.2f} {all_states_translate[1]:.2f} {all_states_translate[2]:.2f}",
            "}",
            "",
            "proc view_state {state_id} {",
            "    setup_msm_representations",
            "    switch -- $state_id {",
        ]
    )

    for ex in exemplars:
        s_id = int(ex["state_id"])
        f_mid: int = int(ex["mid_frame"])
        t_ns: float = float(ex["sim_time_ns"])
        t_us: float = float(ex["sim_time_us"])

        master_script_lines.extend(
            [
                f"        {s_id} {{",
                f"            animate goto {f_mid}",
                f"            apply_state_kap_weights $::state_{s_id}_kap_weights",
                "            apply_custom_colorscale",
                f"            align_view_to_state {effective_rot_x:.1f} {effective_rot_y:.1f} {effective_rot_z:.1f} {vmd_zoom_scale:.2f}",
                "            display update",
                f'            puts ">> Viewing Markov State {s_id}: Frame {f_mid} ({t_ns:.1f} ns / {t_us:.2f} us)"',
                "        }",
            ]
        )

    master_script_lines.extend(
        [
            "        default {",
            f'            puts "Error: Unknown state_id $state_id. Available states: 0 to {len(exemplars)-1}"',
            "        }",
            "    }",
            "}",
            "",
            "proc setup_all_states_representations {} {",
            "    # Remove existing representations",
            "    set nreps [molinfo top get numreps]",
            "    for {set i [expr {$nreps - 1}]} {$i >= 0} {incr i -1} {",
            "        mol delrep $i top",
            "    }",
            "",
            "    # Define color palette (HEAT 5/6 red tones + distinct non-red colors for Markov states)",
            "    color change rgb 1 0.545 0.000 0.000   ;# HEAT 5: Dark Red (#8B0000)",
            "    color change rgb 9 1.000 0.420 0.420   ;# HEAT 6: Light Red (#FF6B6B)",
            "    color change rgb 6 0.470 0.565 0.612   ;# Kap95 Non-HEAT: Slate Gray (#78909C)",
            "    color change rgb 3 1.000 0.549 0.000   ;# Orange (#FF8C00)",
            "    color change rgb 0 0.118 0.533 0.898   ;# Blue (#1E88E5)",
            "    color change rgb 7 0.078 0.612 0.149   ;# Green (#149C26)",
            "    color change rgb 11 0.557 0.141 0.667  ;# Purple (#8E24AA)",
            "    color change rgb 10 0.000 0.675 0.757  ;# Cyan (#00ACC1)",
            "    color change rgb 12 0.486 0.702 0.259  ;# Lime (#7CB342)",
            "    color change rgb 14 1.000 0.702 0.000  ;# Amber (#FFB300)",
            "    color change rgb 13 0.671 0.278 0.737  ;# Mauve (#AB47BC)",
            "    color change rgb 15 0.149 0.776 0.855  ;# Iceblue (#26C6DA)",
            "    color change rgb 5 0.000 0.537 0.482   ;# Teal (#00897B)",
            "",
            "    # Tune material transparency for background Kap95 ribbon (more transparent)",
            "    material change opacity Transparent 0.25",
            "",
            "    # Create dedicated slightly transparent material for multi-state FSFGs with smooth shading",
            '    if {[lsearch [material list] "FSFGTrans"] == -1} {',
            "        material add FSFGTrans copy AOChalky",
            "    }",
            "    material change opacity FSFGTrans 0.80",
            "    material change ambient FSFGTrans 0.25",
            "    material change diffuse FSFGTrans 0.75",
            "    material change specular FSFGTrans 0.10",
            "    material change shininess FSFGTrans 0.20",
            "",
            "    # 1. Native Kap95 Non-HEAT (Frame 0) - Transparent Silver/Slate Ribbon",
            "    mol representation NewCartoon 0.28 10.0 4.1 0",
            "    mol color ColorID 6",
            f'    mol selection "protein and resid 1 to 861 and not (resid {heat5_range[0]} to {heat5_range[1]} or resid {heat6_range[0]} to {heat6_range[1]})"',
            "    mol material Transparent",
            "    mol addrep top",
            "    set kap_rep_idx [expr {[molinfo top get numreps] - 1}]",
            '    mol drawframes top $kap_rep_idx "0"',
            "",
            f"    # 2. Native HEAT Repeat 5 (resid {heat5_range[0]} to {heat5_range[1]}) - Solid Dark Red Ribbon",
            "    mol representation NewCartoon 0.45 10.0 4.1 0",
            "    mol color ColorID 1",
            f'    mol selection "resid {heat5_range[0]} to {heat5_range[1]}"',
            "    mol material Glossy",
            "    mol addrep top",
            "    set h5_rep_idx [expr {[molinfo top get numreps] - 1}]",
            '    mol drawframes top $h5_rep_idx "0"',
            "",
            f"    # 3. Native HEAT Repeat 6 (resid {heat6_range[0]} to {heat6_range[1]}) - Solid Light Red Ribbon",
            "    mol representation NewCartoon 0.45 10.0 4.1 0",
            "    mol color ColorID 9",
            f'    mol selection "resid {heat6_range[0]} to {heat6_range[1]}"',
            "    mol material Glossy",
            "    mol addrep top",
            "    set h6_rep_idx [expr {[molinfo top get numreps] - 1}]",
            '    mol drawframes top $h6_rep_idx "0"',
            "",
        ]
    )

    # Add only the 4 focal FSFG AAs for each Markov state on its exemplar frame (slightly transparent)
    for idx_ex, ex in enumerate(exemplars):
        s_id = int(ex["state_id"])
        f_mid = int(ex["mid_frame"])
        color_id: int = state_color_ids[idx_ex % len(state_color_ids)]

        master_script_lines.extend(
            [
                f"    # State {s_id} (Frame {f_mid}) - Focal FSFG 4 AAs ({focal_names_str}) Backbone Tube (Slightly Transparent)",
                "    mol representation Tube 0.35 16.0",
                f"    mol color ColorID {color_id}",
                f'    mol selection "({focal_res_sel_str})"',
                "    mol material FSFGTrans",
                "    mol addrep top",
                "    set r_idx [expr {[molinfo top get numreps] - 1}]",
                f'    mol drawframes top $r_idx "{f_mid}"',
                "",
                f"    # State {s_id} (Frame {f_mid}) - Focal FSFG 4 AAs Side Chains Licorice (Slightly Transparent)",
                "    mol representation Licorice 0.28 12.0 12.0",
                f"    mol color ColorID {color_id}",
                f'    mol selection "({focal_res_sel_str}) and not hydrogen and (sidechain or name CA)"',
                "    mol material FSFGTrans",
                "    mol addrep top",
                "    set r_idx [expr {[molinfo top get numreps] - 1}]",
                f'    mol drawframes top $r_idx "{f_mid}"',
                "",
            ]
        )

    master_script_lines.extend(
        [
            "    # Display settings",
            "    display projection Orthographic",
            "    display depthcue off",
            "    display backgroundgradient off",
            "    display culling on",
            "    display rendermode GLSL",
            "    color Display Background white",
            "    axes location off",
            "}",
            "",
            "# Apply custom color scale globally on script load",
            "apply_custom_colorscale",
            "",
            "proc view_all_states {} {",
            "    align_all_exemplar_frames",
            "    setup_all_states_representations",
            "    animate goto 0",
            f"    align_view_to_state {effective_rot_x:.1f} {effective_rot_y:.1f} {effective_rot_z:.1f} {vmd_zoom_scale:.2f}",
            '    puts ">> Viewing All Markov States Overlay on Native Kap95 Structure"',
            "}",
            "",
            "proc render_state_png {filename} {",
            "    set base_no_ext [file rootname $filename]",
            "    set tmp_tga \"${base_no_ext}.tmp.tga\"",
            "    render snapshot $tmp_tga",
            "    display update",
            "",
            "    # Convert raw TGA framebuffer capture to true RFC-standard PNG using Python PIL",
            "    set py_script \"from PIL import Image; im = Image.open(r'$tmp_tga'); im.save(r'$filename', format='PNG')\"",
            "    set converted 0",
            "    if {[catch {exec conda run -n general python3 -c $py_script} py3_err] == 0} {",
            "        set converted 1",
            "    } elseif {[catch {exec python -c $py_script} py_err] == 0} {",
            "        set converted 1",
            "    } elseif {[catch {exec convert $tmp_tga $filename} conv_err] == 0} {",
            "        set converted 1",
            "    }",
            "",
            "    if {$converted} {",
            "        file delete -force $tmp_tga",
            "    } else {",
            "        file rename -force $tmp_tga \"${base_no_ext}.tga\"",
            "        puts \">> Notice: Saved raw snapshot as ${base_no_ext}.tga\"",
            "    }",
            "}",
            "",
            "proc render_all_states {out_dir} {",
            "    file mkdir $out_dir",
            "    align_all_exemplar_frames",
            f"    foreach s {{{' '.join(str(e['state_id']) for e in exemplars)}}} {{",
            "        view_state $s",
            "        display update",
            "        after 300",
            '        set filename [format "%s/state_%d.png" $out_dir $s]',
            "        render_state_png $filename",
            '        puts ">> Rendered and saved: $filename"',
            "    }",
            "    # Render 1 additional image showing all states at once on native structure",
            "    view_all_states",
            "    display update",
            "    after 300",
            '    set all_filename [format "%s/all_states.png" $out_dir]',
            "    render_state_png $all_filename",
            '    puts ">> Rendered and saved all states overlay: $all_filename"',
            '    puts ">> Done rendering all states!"',
            "}",
            "",
            "# Apply display and background settings immediately on load",
            "display projection Orthographic",
            "display depthcue off",
            "display backgroundgradient off",
            "display culling on",
            "display rendermode GLSL",
            "color Display Background white",
            "axes location off",
            "",
            'puts "================================================================="',
            'puts "iMSM VMD State Visualization Tools Loaded!"',
            'puts "  - Use: view_state <state_id>   (e.g., view_state 0)"',
            'puts "  - Use: view_all_states         (Overlay all states on native Kap95)"',
            f'puts "  - Use: render_all_states \\"{snapshots_dir}\\""',
            'puts "================================================================="',
        ]
    )

    master_script_path: str = os.path.join(output_dir, "master_states.vmd")
    with open(master_script_path, "w") as f_out:
        f_out.write("\n".join(master_script_lines) + "\n")

    # Generate standalone .vmd file for each state
    for ex in exemplars:
        s_id = int(ex["state_id"])
        f_mid = int(ex["mid_frame"])
        t_ns = float(ex["sim_time_ns"])
        t_us = float(ex["sim_time_us"])

        single_state_lines: list[str] = [
            f"# Standalone VMD script for Markov State {s_id}",
            f"# Simulation Frame: {f_mid} ({t_ns:.1f} ns / {t_us:.2f} us)",
            "",
            f"source {os.path.abspath(master_script_path)}",
            f"view_state {s_id}",
            f'render_state_png "{snapshots_dir}/state_{s_id}.png"',
        ]
        state_script_path: str = os.path.join(output_dir, f"state_{s_id}.vmd")
        with open(state_script_path, "w") as f_state:
            f_state.write("\n".join(single_state_lines) + "\n")

    # Generate standalone .vmd file for all states overlay
    all_states_script_lines: list[str] = [
        "# Standalone VMD script for All Markov States Overlay on Native Kap95",
        "",
        f"source {os.path.abspath(master_script_path)}",
        "view_all_states",
        f'render_state_png "{snapshots_dir}/all_states.png"',
    ]
    all_states_script_path: str = os.path.join(output_dir, "all_states.vmd")
    with open(all_states_script_path, "w") as f_all:
        f_all.write("\n".join(all_states_script_lines) + "\n")

    print("\n=================================================================")
    print(f"Generated VMD State Visualizations in: {output_dir}")
    print(f"Master script: {master_script_path}")
    print(f"All-states script: {all_states_script_path}")
    print("-----------------------------------------------------------------")
    print(f"{'State':<7}{'Mid Frame':<12}{'Sim Time (ns)':<16}{'Sim Time (µs)':<16}{'Centroid Dist':<16}")
    print("-----------------------------------------------------------------")
    for ex in exemplars:
        print(
            f"{int(ex['state_id']):<7}{int(ex['mid_frame']):<12}{float(ex['sim_time_ns']):<16.1f}{float(ex['sim_time_us']):<16.2f}{float(ex['centroid_distance']):<16.4f}"
        )
    print("=================================================================\n")

    return exemplars


def generate_fg_vmd_scripts(
    top_path: str,
    traj_paths: list[str],
    imsm_checkpoint_path: str,
    output_dir: str,
    kap_n_ca: int,
    focal_fg_alphacarbon: int | tuple[int, int] | list[int] | str,
    window_size: int,
    stride: int,
    first_frame: int | None,
    ns_per_frame: float,
    free_threshold: float,
    auto_align_plane: bool,
    rotation_x_deg: float,
    rotation_y_deg: float,
    rotation_z_deg: float,
    heat5_range: tuple[int, int],
    heat6_range: tuple[int, int],
    vmd_zoom_scale: float,
    all_states_translate: tuple[float, float, float],
) -> list[dict[str, int | float]]:
    """Generate VMD Tcl scripts and snapshots for Markov states (Panels A & C)."""
    return generate_vmd_state_scripts(
        top_path=top_path,
        traj_paths=traj_paths,
        imsm_checkpoint_path=imsm_checkpoint_path,
        output_dir=output_dir,
        kap_n_ca=kap_n_ca,
        focal_fg_alphacarbon=focal_fg_alphacarbon,
        window_size=window_size,
        stride=stride,
        first_frame=first_frame,
        ns_per_frame=ns_per_frame,
        free_threshold=free_threshold,
        auto_align_plane=auto_align_plane,
        rotation_x_deg=rotation_x_deg,
        rotation_y_deg=rotation_y_deg,
        rotation_z_deg=rotation_z_deg,
        heat5_range=heat5_range,
        heat6_range=heat6_range,
        vmd_zoom_scale=vmd_zoom_scale,
        all_states_translate=all_states_translate,
    )


def hex_to_rgb_float(hex_code: str) -> tuple[float, float, float]:
    """Convert hex color string to normalized RGB float tuple in range [0.0, 1.0]."""
    clean_hex: str = hex_code.lstrip("#")
    r_int: int = int(clean_hex[0:2], 16)
    g_int: int = int(clean_hex[2:4], 16)
    b_int: int = int(clean_hex[4:6], 16)
    return float(r_int / 255.0), float(g_int / 255.0), float(b_int / 255.0)


def generate_fg_overview_vmd_script(
    top_path: str,
    traj_paths: list[str],
    output_dir: str,
    output_filename: str,
    kap_n_ca: int,
    focal_fg_alphacarbon: int | tuple[int, int] | list[int] | str,
    frame_idx: int | None,
    auto_align_plane: bool,
    rotation_x_deg: float,
    rotation_y_deg: float,
    rotation_z_deg: float,
    heat5_range: tuple[int, int],
    heat6_range: tuple[int, int],
    vmd_zoom_scale: float,
    kap_color_hex: str,
    fsfg_color_hex: str,
) -> str:
    """Generate a self-contained, portable VMD script showing the full Kap95 at the simulation midpoint."""
    pdb: md.Trajectory = md.load(top_path)
    all_ca: np.ndarray = pdb.topology.select("name CA")
    kap_ca: np.ndarray = all_ca[:kap_n_ca]
    fg1_ca: np.ndarray = all_ca[kap_n_ca : kap_n_ca + 125]

    focal_indices, focal_label = extract_focal_info(focal_fg_alphacarbon=focal_fg_alphacarbon)
    focal_ca_indices: list[int] = [int(fg1_ca[idx]) for idx in focal_indices]
    focal_atoms = [pdb.topology.atom(idx) for idx in focal_ca_indices]
    focal_res_seqs: list[int] = [int(atom.residue.resSeq) for atom in focal_atoms]
    focal_res_sel_str: str = " or ".join([f"resid {s}" for s in sorted(set(focal_res_seqs))])
    focal_names_str: str = "+".join([f"{atom.residue.name}{atom.residue.resSeq}" for atom in focal_atoms])

    kap_xyz_angstrom: np.ndarray = pdb.xyz[0, kap_ca, :] * 10.0

    effective_rot_x: float = rotation_x_deg
    effective_rot_y: float = rotation_y_deg
    effective_rot_z: float = rotation_z_deg

    if auto_align_plane:
        _, effective_rot_x, effective_rot_y, effective_rot_z = (
            compute_heat_repeats_alignment_matrix(
                kap_xyz=kap_xyz_angstrom / 10.0,
                heat5_range=heat5_range,
                heat6_range=heat6_range,
            )
        )

    kap_r, kap_g, kap_b = hex_to_rgb_float(hex_code=kap_color_hex)
    fsfg_r, fsfg_g, fsfg_b = hex_to_rgb_float(hex_code=fsfg_color_hex)

    os.makedirs(output_dir, exist_ok=True)
    out_script_path: str = os.path.join(output_dir, output_filename)

    fg1_min_res: int = kap_n_ca + 1
    fg1_max_res: int = kap_n_ca + 125

    frame_init_line: str = (
        f"set target_frame {frame_idx}"
        if frame_idx is not None
        else (
            "if {![info exists target_frame]} {\n"
            "    set num_f [molinfo top get numframes]\n"
            "    if {$num_f > 1} {\n"
            "        set target_frame [expr {$num_f / 2}]\n"
            "    } else {\n"
            "        set target_frame 0\n"
            "    }\n"
            "}"
        )
    )

    tcl_lines: list[str] = [
        "# ==============================================================================",
        "# Standalone VMD Script: Full Kap95 Simulation Overview (Panel Extra)",
        f"# Focal Component: {focal_label} ({focal_names_str})",
        f"# Kap95 Color: {kap_color_hex} | FSFG Color: {fsfg_color_hex}",
        "# ==============================================================================",
        "",
        "# Determine target frame (midpoint of trajectory unless pre-defined)",
        frame_init_line,
        "",
        "proc setup_full_kap_overview_representations {} {",
        "    # Remove existing representations",
        "    set nreps [molinfo top get numreps]",
        "    for {set i [expr {$nreps - 1}]} {$i >= 0} {incr i -1} {",
        "        mol delrep $i top",
        "    }",
        "",
        "    # Custom color palette",
        f"    color change rgb 30 {kap_r:.4f} {kap_g:.4f} {kap_b:.4f}   ;# Kap95: {kap_color_hex}",
        f"    color change rgb 31 {fsfg_r:.4f} {fsfg_g:.4f} {fsfg_b:.4f}   ;# FSFG: {fsfg_color_hex}",
        "    color change rgb 32 0.750 0.750 0.750                  ;# FG Chain 1 non-focal: Light Gray",
        "",
        "    # 1. Whole Kap95 - Softly shaded (AOChalky) NewCartoon",
        "    mol representation NewCartoon 0.35 10.0 4.1 0",
        "    mol color ColorID 30",
        f'    mol selection "protein and resid 1 to {kap_n_ca}"',
        "    mol material AOChalky",
        "    mol addrep top",
        "",
        f"    # 2. Focal FSFG 4 AAs ({focal_names_str}) Backbone - Softly shaded Tube",
        "    mol representation Tube 0.35 16.0",
        "    mol color ColorID 31",
        f'    mol selection "({focal_res_sel_str})"',
        "    mol material AOChalky",
        "    mol addrep top",
        "",
        f"    # 3. Focal FSFG 4 AAs ({focal_names_str}) Sidechains - Softly shaded Licorice",
        "    mol representation Licorice 0.28 12.0 12.0",
        "    mol color ColorID 31",
        f'    mol selection "({focal_res_sel_str}) and not hydrogen and (sidechain or name CA)"',
        "    mol material AOChalky",
        "    mol addrep top",
        "",
        "    # 4. Non-focal FG Chain 1 - Solid Full-Color Tube",
        "    mol representation Tube 0.22 16.0",
        "    mol color ColorID 31",
        f'    mol selection "resid {fg1_min_res} to {fg1_max_res} and not ({focal_res_sel_str})"',
        "    mol material AOChalky",
        "    mol addrep top",
        "",
        "    # Display settings matching publication style",
        "    display projection Orthographic",
        "    display depthcue off",
        "    display backgroundgradient off",
        "    display culling on",
        "    display rendermode GLSL",
        "    color Display Background white",
        "    axes location off",
        "}",
        "",
        "proc align_and_center_view {f_idx rot_x rot_y rot_z zoom_scale} {",
        f'    set sel_kap [atomselect top "protein and resid 1 to {kap_n_ca}" frame $f_idx]',
        "    set kap_center [measure center $sel_kap]",
        "    $sel_kap delete",
        "    set shift [vecscale -1.0 $kap_center]",
        '    set move_all [atomselect top "all" frame $f_idx]',
        "    $move_all moveby $shift",
        "    $move_all delete",
        "",
        "    display resetview",
        "    translate to 0.0 0.0 0.0",
        "    molinfo top set center [list {0.0 0.0 0.0}]",
        "    rotate y by $rot_y",
        "    rotate x by $rot_x",
        "    rotate z by $rot_z",
        "    scale by $zoom_scale",
        "}",
        "",
        "proc view_overview {{f_idx \"\"}} {",
        "    if {$f_idx eq \"\"} {",
        "        set f_idx $::target_frame",
        "    }",
        "    animate goto $f_idx",
        "    setup_full_kap_overview_representations",
        f"    align_and_center_view $f_idx {effective_rot_x:.1f} {effective_rot_y:.1f} {effective_rot_z:.1f} {vmd_zoom_scale:.2f}",
        "    display update",
        '    puts ">> Viewing Full Kap95 Simulation Overview (Frame $f_idx)"',
        "}",
        "",
        "proc render_overview_png {filename} {",
        "    set base_no_ext [file rootname $filename]",
        "    set tmp_tga \"${base_no_ext}.tmp.tga\"",
        "    render snapshot $tmp_tga",
        "    display update",
        "",
        "    # Convert raw TGA framebuffer capture to true RFC-standard PNG using Python PIL",
        "    set py_script \"from PIL import Image; im = Image.open(r'$tmp_tga'); im.save(r'$filename', format='PNG')\"",
        "    set converted 0",
        "    if {[catch {exec conda run -n general_notebooks python -c $py_script} py_err1] == 0} {",
        "        set converted 1",
        "    } elseif {[catch {exec conda run -n general python3 -c $py_script} py_err2] == 0} {",
        "        set converted 1",
        "    } elseif {[catch {exec python3 -c $py_script} py_err3] == 0} {",
        "        set converted 1",
        "    } elseif {[catch {exec python -c $py_script} py_err4] == 0} {",
        "        set converted 1",
        "    } elseif {[catch {exec convert $tmp_tga $filename} conv_err] == 0} {",
        "        set converted 1",
        "    }",
        "",
        "    if {$converted} {",
        "        file delete -force $tmp_tga",
        '        puts ">> Rendered and converted: $filename"',
        "    } else {",
        "        file rename -force $tmp_tga \"${base_no_ext}.tga\"",
        '        puts ">> Notice: Saved raw snapshot as ${base_no_ext}.tga"',
        "    }",
        "}",
        "",
        "# Initialize overview view",
        "view_overview",
        "",
        "# Save snapshot in relative snapshots folder",
        "set script_dir [file dirname [file normalize [info script]]]",
        'set snapshot_dir [file join $script_dir "snapshots"]',
        "file mkdir $snapshot_dir",
        'set snapshot_file [file join $snapshot_dir "full_kap_overview.png"]',
        "render_overview_png $snapshot_file",
        "",
        'puts "================================================================="',
        'puts "Full Kap95 Simulation Overview Loaded!"',
        'puts "  - Use: view_overview <frame_idx>   (Re-orient / jump to frame)"',
        'puts "  - Use: render_overview_png <path>  (Save PNG snapshot)"',
        'puts "================================================================="',
    ]

    with open(out_script_path, "w") as f_out:
        f_out.write("\n".join(tcl_lines) + "\n")

    print(f">> Generated Full Kap95 VMD Overview Script: {out_script_path}")
    return out_script_path

