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
import subprocess
from typing import TypedDict
import matplotlib.patheffects as pe
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.ticker import MaxNLocator
import matplotlib.animation as anim
import mdtraj as md
import numpy as np
from scipy.interpolate import CubicSpline

from figure_fg_sliding_helpers import (
    compute_heat_repeats_alignment_matrix,
    compute_native_states_and_profile,
    create_wedge_marker,
    extract_focal_info,
)
from iMSM.extensions.npc.npc_utils import infinitesimal_generator, radius_a_to_kda
from iMSM.extensions.npc.npc_graph_figure import (
    PIE_COLORS,
    add_npc_scaffold_picture,
    adjust_nucleus_cytoplasm_mus,
    calc_non_spoke_cluster_makeup,
    estimate_cluters_mu_cov,
    estimate_unprojected_mus,
    hide_axii_and_show_scale_bar,
    load_cluster_data,
    pick_good_clusters_by_mu_angle,
    visualize_arrows_between_mesostates,
    visualize_pie_mesostates,
)


class NPCTrajectoryStep(TypedDict):
    """Trajectory step representing transition between mesostates in the NPC."""

    walk_index: int
    step_in_walk: int
    global_step: int
    from_local_idx: int
    to_local_idx: int
    from_cluster_id: int
    to_cluster_id: int
    from_pos: tuple[float, float]
    to_pos: tuple[float, float]
    direction: str
    sim_time_us: float


class NPCFrameState(TypedDict):
    """Visual state for a single video frame in the NPC animation."""

    frame_idx: int
    walk_index: int
    direction: str
    current_cluster_id: int
    sim_time_us: float
    marker_x: float
    marker_y: float
    is_dwelling: bool
    progress_fraction: float
    active_edge_x: list[float]
    active_edge_y: list[float]


class StateTimelineEntry(TypedDict):
    """Timeline entry specifying Markov state status for a specific video frame."""

    video_frame_idx: int
    sim_frame_idx: int
    sim_time_ns: float
    sim_time_us: float
    window_idx: int
    state_id: int
    prev_state_id: int
    is_transition: bool
    transition_from: int
    transition_to: int
    transition_progress: float
    progress_fraction: float


def extract_trajectory_state_timeline(
    imsm_checkpoint_path: str,
    total_frames: int,
    window_size: int,
    stride: int,
    ns_per_frame: float,
    transition_glide_frames: int,
) -> list[StateTimelineEntry]:
    """Build a frame-by-frame state timeline mapping simulation frames to active Markov states."""
    clustering_pickle_path: str = os.path.join(imsm_checkpoint_path, "3_clustering.pickle")
    if not os.path.exists(clustering_pickle_path):
        raise FileNotFoundError(
            f"Clustering pickle not found at: {clustering_pickle_path}. Check imsm_checkpoint_path."
        )

    with open(clustering_pickle_path, "rb") as f_cluster:
        imsm_cluster = pickle.load(f_cluster)

    discrete_trajectories: np.ndarray = np.array(imsm_cluster.trajectories[0].trajectory)
    # Shape is typically (1, n_windows) or (n_windows,)
    if discrete_trajectories.ndim == 2:
        state_sequence: np.ndarray = discrete_trajectories[0]
    else:
        state_sequence = discrete_trajectories

    n_windows: int = int(len(state_sequence))
    frame_indices: list[int] = list(range(0, total_frames, stride))
    total_video_frames: int = len(frame_indices)
    frames_per_window: int = max(1, window_size // stride)

    timeline: list[StateTimelineEntry] = []

    for v_idx, f_idx in enumerate(frame_indices):
        w_idx: int = min(f_idx // window_size, n_windows - 1)
        target_state: int = int(state_sequence[w_idx])
        prev_w_state: int = int(state_sequence[w_idx - 1]) if w_idx > 0 else target_state
        time_ns: float = f_idx * ns_per_frame
        time_us: float = time_ns / 1000.0

        window_start_v_idx: int = w_idx * frames_per_window
        offset_in_window: int = v_idx - window_start_v_idx

        is_trans: bool = (target_state != prev_w_state) and (offset_in_window < transition_glide_frames)
        trans_from: int
        trans_to: int
        trans_prog: float
        curr_state: int

        if is_trans:
            trans_from = prev_w_state
            trans_to = target_state
            alpha: float = float(offset_in_window + 1) / float(transition_glide_frames)
            trans_prog = 0.5 * (1.0 - float(np.cos(np.pi * alpha)))
            curr_state = trans_to if alpha >= 0.5 else trans_from
        else:
            trans_from = target_state
            trans_to = target_state
            trans_prog = 0.0
            curr_state = target_state

        progress: float = float(v_idx) / float(max(1, total_video_frames - 1))

        timeline.append(
            StateTimelineEntry(
                video_frame_idx=v_idx,
                sim_frame_idx=f_idx,
                sim_time_ns=time_ns,
                sim_time_us=time_us,
                window_idx=w_idx,
                state_id=curr_state,
                prev_state_id=prev_w_state,
                is_transition=is_trans,
                transition_from=trans_from,
                transition_to=trans_to,
                transition_progress=trans_prog,
                progress_fraction=progress,
            )
        )

    return timeline


def compute_log_state_weights(
    imsm_checkpoint_path: str,
    kap_n_ca: int,
) -> np.ndarray:
    """Compute normalized log-scaled Kap95 interaction weights for each Markov state."""
    clustering_pickle_path: str = os.path.join(imsm_checkpoint_path, "3_clustering.pickle")
    with open(clustering_pickle_path, "rb") as f_cluster:
        imsm_cluster = pickle.load(f_cluster)

    cluster_centers: np.ndarray = np.array(imsm_cluster.clustering.cluster_centers_)
    comps: np.ndarray = np.array(imsm_cluster.unique_components)
    n_states: int = cluster_centers.shape[0]

    kap_weights: np.ndarray = np.zeros((n_states, kap_n_ca), dtype=float)
    for col_idx, comp_name in enumerate(comps):
        if str(comp_name).startswith("kapC"):
            res_idx: int = int(str(comp_name)[4:])
            if res_idx < kap_n_ca:
                kap_weights[:, res_idx] = cluster_centers[:, col_idx]

    log_weights: np.ndarray = np.zeros_like(kap_weights)
    valid_mask: np.ndarray = kap_weights >= 0.001
    log_weights[valid_mask] = np.clip((np.log10(kap_weights[valid_mask]) + 3.0) / 3.0, 0.0, 1.0)
    return log_weights


def generate_vmd_movie_script(
    top_path: str,
    traj_paths: list[str],
    imsm_checkpoint_path: str,
    output_dir: str,
    output_movie_path: str,
    kap_n_ca: int,
    focal_fg_alphacarbon: int | tuple[int, int] | list[int] | str,
    window_size: int,
    stride: int,
    fps: int,
    ns_per_frame: float,
    auto_align_plane: bool,
    rotation_x_deg: float,
    rotation_y_deg: float,
    rotation_z_deg: float,
    heat5_range: tuple[int, int],
    heat6_range: tuple[int, int],
    vmd_zoom_scale: float,
    all_states_translate: tuple[float, float, float],
    width: int,
    height: int,
    smoothing_window_ns: float,
) -> str:
    """Generate VMD Tcl script to render a synchronized 3D trajectory video matching Figure 6."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_movie_path)), exist_ok=True)

    smoothing_frames: int = (
        max(0, int(round((smoothing_window_ns / ns_per_frame) / 2.0)))
        if smoothing_window_ns > 0.0
        else 0
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

    log_weights: np.ndarray = compute_log_state_weights(
        imsm_checkpoint_path=imsm_checkpoint_path,
        kap_n_ca=kap_n_ca,
    )
    n_states: int = log_weights.shape[0]

    clustering_pickle_path: str = os.path.join(imsm_checkpoint_path, "3_clustering.pickle")
    with open(clustering_pickle_path, "rb") as f_cluster:
        imsm_cluster = pickle.load(f_cluster)
    discrete_trajectories: np.ndarray = np.array(imsm_cluster.trajectories[0].trajectory)
    state_seq_arr: np.ndarray = (
        discrete_trajectories[0] if discrete_trajectories.ndim == 2 else discrete_trajectories
    )
    state_seq_tcl_str: str = " ".join(str(int(s)) for s in state_seq_arr)

    # 1024-step color scale matching Figure 6 (Slate Gray #78909C -> Red #D32F2F)
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

    script_lines: list[str] = [
        "# ==============================================================================",
        "# VMD Trajectory Video Generator for Figure 6 Synchronization",
        f"# Focal Component: {focal_label} ({focal_names_str})",
        "# ==============================================================================",
        "",
        f'set output_movie "{os.path.basename(output_movie_path)}"',
        f"set frame_stride {stride}",
        f"set video_fps {fps}",
        f"set window_size {window_size}",
        f"set ns_per_frame {ns_per_frame}",
        f"set smoothing_window_ns {smoothing_window_ns}",
        f"set smoothing_frames {smoothing_frames}",
        f"set target_width {width}",
        f"set target_height {height}",
        "",
        "# Per-state log-scaled interaction weights on Kap95",
    ]

    for s_idx in range(n_states):
        w_str: str = " ".join(f"{float(w):.4f}" for w in log_weights[s_idx, :])
        script_lines.append(f"set state_{s_idx}_kap_weights [list {w_str}]")

    script_lines.extend(
        [
            "",
            f"set window_state_sequence [list {state_seq_tcl_str}]",
            "set n_windows [llength $window_state_sequence]",
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
            "    # Apply beta weights to the current frame",
            "    set cur_f [molinfo top get frame]",
            '    set sel_f [atomselect top "protein and resid 1 to 861" frame $cur_f]',
            "    $sel_f set beta $beta_vals",
            "    $sel_f delete",
            "}",
            "",
        ]
    )

    script_lines.extend(color_scale_lines)

    script_lines.extend(
        [
            "",
            "set frames_aligned 0",
            "proc align_all_frames_to_heat {} {",
            "    if {$::frames_aligned} {",
            "        return",
            "    }",
            '    puts ">> Aligning all trajectory frames to reference frame 0 via HEAT repeats 5 & 6..."',
            f'    set sel_ref [atomselect top "protein and (resid {heat5_range[0]} to {heat5_range[1]} or resid {heat6_range[0]} to {heat6_range[1]}) and name CA" frame 0]',
            "    set num_frames [molinfo top get numframes]",
            "    for {set f 1} {$f < $num_frames} {incr f} {",
            f'        set sel_cur [atomselect top "protein and (resid {heat5_range[0]} to {heat5_range[1]} or resid {heat6_range[0]} to {heat6_range[1]}) and name CA" frame $f]',
            "        set trans_mat [measure fit $sel_cur $sel_ref]",
            '        set move_all [atomselect top "all" frame $f]',
            "        $move_all move $trans_mat",
            "        $sel_cur delete",
            "        $move_all delete",
            "    }",
            "    set heat_center [measure center $sel_ref]",
            "    set shift [vecscale -1.0 $heat_center]",
            "    for {set f 0} {$f < $num_frames} {incr f} {",
            '        set move_all [atomselect top "all" frame $f]',
            "        $move_all moveby $shift",
            "        $move_all delete",
            "    }",
            "    $sel_ref delete",
            "    set ::frames_aligned 1",
            '    puts ">> Frame alignment complete."',
            "}",
            "",
            "proc setup_video_representations {} {",
            "    set nreps [molinfo top get numreps]",
            "    for {set i [expr {$nreps - 1}]} {$i >= 0} {incr i -1} {",
            "        mol delrep $i top",
            "    }",
            "",
            "    # Color definitions matching 2D network plot",
            "    color change rgb 7 0.078 0.612 0.149   ;# Focal FSFG Motif: Green (#149C26)",
            "    color change rgb 2 0.750 0.750 0.750   ;# FG Chain 2: Light Gray",
            "    color change rgb 8 1.000 1.000 1.000   ;# Background: Pure White",
            "",
            "    material change opacity Transparent 0.25",
            "",
            f"    # 1. Background Kap95 Non-HEAT Ribbon - Semi-Transparent Colored by Interaction Beta",
            "    mol representation NewCartoon 0.28 10.0 4.1 0",
            "    mol color Beta",
            f'    mol selection "protein and resid 1 to 861 and not (resid {heat5_range[0]} to {heat5_range[1]} or resid {heat6_range[0]} to {heat6_range[1]})"',
            "    mol material Transparent",
            "    mol addrep top",
            "    set kap_rep_idx [expr {[molinfo top get numreps] - 1}]",
            "    mol scaleminmax top $kap_rep_idx 0.0 1.0",
            "",
            f"    # 2. HEAT Repeats 5 & 6 (resid {heat5_range[0]} to {heat5_range[1]} & {heat6_range[0]} to {heat6_range[1]}) - OPAQUE Solid Ribbon Colored by Beta",
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
            f"    # 4. Focal FSFG 4 AAs ({focal_names_str}) Side Chains - Green Licorice",
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
            "    apply_custom_colorscale",
            "",
            "    if {$::smoothing_frames > 0} {",
            "        apply_smoothing $::smoothing_frames",
            "    }",
            "",
            "    # Camera projection matching 2D network plot",
            "    display projection Orthographic",
            "    display depthcue off",
            "    display backgroundgradient off",
            "    color Display Background white",
            "    axes location off",
            "    display resetview",
            "    translate to 0.0 0.0 0.0",
            "    molinfo top set center [list {0.0 0.0 0.0}]",
            f"    rotate y by {effective_rot_y:.1f}",
            f"    rotate x by {effective_rot_x:.1f}",
            f"    rotate z by {effective_rot_z:.1f}",
            f"    scale by {vmd_zoom_scale:.2f}",
            f"    translate by {all_states_translate[0]:.2f} {all_states_translate[1]:.2f} {all_states_translate[2]:.2f}",
            "}",
            "",
            "proc apply_smoothing {{n -1}} {",
            "    if {$n < 0} { set n $::smoothing_frames }",
            "    if {$n < 0} { set n 0 }",
            "    set num_reps [molinfo top get numreps]",
            "    for {set r 0} {$r < $num_reps} {incr r} {",
            "        mol smoothrep top $r $n",
            "    }",
            "    set total_frames [expr {2 * $n + 1}]",
            "    set total_ns [expr {$total_frames * $::ns_per_frame}]",
            "    puts [format \">> Applied trajectory smoothing: %d frames (+/- %.1f ns, total moving average window: %.1f ns over %d frames)\" $n [expr {$n * $::ns_per_frame}] $total_ns $total_frames]",
            "}",
            "",
            "proc render_simulation_movie {{out_mp4 \"\"} {step_stride -1} {fps -1}} {",
            "    if {$out_mp4 eq \"\"} { set out_mp4 $::output_movie }",
            "    if {$step_stride <= 0} { set step_stride $::frame_stride }",
            "    if {$fps <= 0} { set fps $::video_fps }",
            "",
            "    set temp_frames_dir [file join [file dirname $out_mp4] \"temp_vmd_frames\"]",
            "    file mkdir $temp_frames_dir",
            "",
            "    set total_sim_frames [molinfo top get numframes]",
            "    set last_state -1",
            "    set frame_count 0",
            "",
            '    puts ">> Starting VMD simulation frame rendering..."',
            '    puts "   Total frames: $total_sim_frames, Stride: $step_stride, Target FPS: $fps"',
            "",
            "    for {set f 0} {$f < $total_sim_frames} {incr f $step_stride} {",
            "        animate goto $f",
            "        display update",
            "",
            "        set w [expr {$f / $::window_size}]",
            "        if {$w >= $::n_windows} { set w [expr {$::n_windows - 1}] }",
            "        set cur_state [lindex $::window_state_sequence $w]",
            "",
            "        if {$cur_state != $last_state} {",
            "            set cur_weights [set ::state_${cur_state}_kap_weights]",
            "            apply_state_kap_weights $cur_weights",
            "            apply_custom_colorscale",
            "            display update",
            "            set last_state $cur_state",
            "        }",
            "",
            '        set img_path [format "%s/frame_%06d.tga" $temp_frames_dir $frame_count]',
            "        render snapshot $img_path",
            "        incr frame_count",
            "    }",
            "",
            '    puts ">> Rendered $frame_count frames. Compiling MP4 via ffmpeg..."',
            '    set ffmpeg_cmd [format "ffmpeg -y -framerate %d -i \\"%s/frame_%%06d.tga\\" -c:v libx264 -crf 18 -pix_fmt yuv420p \\"%s\\"" $fps $temp_frames_dir $out_mp4]',
            '    set catch_res [catch {exec {*}$ffmpeg_cmd 2>@1} ff_out]',
            '    if {$catch_res == 0 || [file exists $out_mp4]} {',
            '        puts ">> Successfully created VMD simulation video: $out_mp4"',
            '        file delete -force $temp_frames_dir',
            '    } else {',
            '        puts ">> ffmpeg failed or error occurred: $ff_out"',
            '        puts "   Temporary frames kept in: $temp_frames_dir"',
            '    }',
            "}",
            "",
            "proc init_and_prepare {} {",
            "    if {[molinfo num] == 0} {",
            '        puts ">> Notice: No molecule loaded in VMD yet."',
            '        puts "   Please load your structure and trajectory first, then run: init_and_prepare"',
            "        return",
            "    }",
            "    align_all_frames_to_heat",
            "    setup_video_representations",
            '    puts "================================================================="',
            '    puts "VMD Trajectory Video Environment Ready!"',
            "    if {$::smoothing_frames > 0} {",
            "        puts [format \">> Trajectory smoothing active: %d frames (+/- %.1f ns, ~%.1f ns moving average window)\" $::smoothing_frames [expr {$::smoothing_frames * $::ns_per_frame}] [expr {(2 * $::smoothing_frames + 1) * $::ns_per_frame}]]",
            "        puts \"   (You can adjust smoothing at any time by running: apply_smoothing <n_frames>)\"",
            "    }",
            '    puts "Run: render_simulation_movie to generate the video."',
            '    puts "================================================================="',
            "}",
            "",
            "# Apply display and background settings immediately on script load",
            "display projection Orthographic",
            "display depthcue off",
            "display backgroundgradient off",
            "display culling on",
            "display rendermode GLSL",
            "color Display Background white",
            "axes location off",
            "",
            "if {[molinfo num] > 0} {",
            "    init_and_prepare",
            "} else {",
            '    puts "================================================================="',
            '    puts "VMD Trajectory Video Script Loaded!"',
            '    puts "Load your structure & trajectory into VMD, then run: init_and_prepare"',
            '    puts "================================================================="',
            "}",
        ]
    )

    script_path: str = os.path.join(output_dir, "render_trajectory_movie.vmd")
    with open(script_path, "w") as f_out:
        f_out.write("\n".join(script_lines) + "\n")

    return script_path


def generate_imsm_network_video(
    top_path: str,
    imsm_checkpoint_path: str,
    output_video_path: str,
    total_frames: int,
    stride: int,
    window_size: int,
    fps: int,
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
    heat_ribbon_width: float,
    heat_ribbon_alpha: float,
    state_color_mode: str,
    arrow_color: str | None,
    show_rate_colorbar: bool,
    show_residue_colorbar: bool,
    diverging_palette: tuple[str, ...],
    residue_color_range: tuple[int, int],
    ns_per_frame: float,
    dpi: int,
    transition_glide_frames: int,
) -> str:
    """Generate animated MP4 of the 2D iMSM network matching Figure 6 Panel D with active states and transitions highlighted."""
    timeline: list[StateTimelineEntry] = extract_trajectory_state_timeline(
        imsm_checkpoint_path=imsm_checkpoint_path,
        total_frames=total_frames,
        window_size=window_size,
        stride=stride,
        ns_per_frame=ns_per_frame,
        transition_glide_frames=transition_glide_frames,
    )

    pdb: md.Trajectory = md.load(top_path)
    all_ca: np.ndarray = pdb.topology.select("name CA")
    kap_ca: np.ndarray = all_ca[:kap_n_ca]
    kap_xyz: np.ndarray = pdb.xyz[0, kap_ca, :]

    with open(f"{imsm_checkpoint_path}/3_clustering.pickle", "rb") as f_cluster:
        imsm_cluster = pickle.load(f_cluster)
    cluster_centers: np.ndarray = np.array(imsm_cluster.clustering.cluster_centers_)
    comps: np.ndarray = np.array(imsm_cluster.unique_components)

    with open(f"{imsm_checkpoint_path}/4_msm.pickle", "rb") as f_msm:
        imsm_msm = pickle.load(f_msm)
    msm_mat: np.ndarray = imsm_msm.transition_matrix
    n_clusters: int = msm_mat.shape[0]

    dt_us: float = dt_ns / 1000.0
    generator_mat: np.ndarray = infinitesimal_generator(msm_mat, dt=dt_us)
    rate_mat: np.ndarray = generator_mat.copy()
    np.fill_diagonal(rate_mat, 0.0)
    rate_mat = np.maximum(rate_mat, 0.0)

    unbound_matches: np.ndarray = np.array(
        [i for i, c in enumerate(comps) if str(c).strip().lower() == "unbound"], dtype=int
    )
    unbound_col_idx: int = int(unbound_matches[0]) if len(unbound_matches) > 0 else -1
    unbound_weights: np.ndarray = (
        cluster_centers[:, unbound_col_idx] if unbound_col_idx >= 0 else np.zeros(n_clusters, dtype=float)
    )
    candidate_free_idx: int = int(np.argmax(unbound_weights)) if len(unbound_weights) > 0 else 0
    free_state_idx: int | None = (
        candidate_free_idx
        if (len(unbound_weights) > 0 and float(unbound_weights[candidate_free_idx]) >= free_threshold)
        else None
    )

    native_state_indices, _, _ = compute_native_states_and_profile(
        native_pdb_path=native_pdb_path,
        kap_n_ca=kap_n_ca,
        focal_fg_alphacarbon=focal_fg_alphacarbon,
        interaction_capacity=interaction_capacity,
        max_surface_dist=max_surface_dist,
        unique_components=comps,
        cluster_centers=cluster_centers,
        similarity_threshold=native_similarity_threshold,
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

    rot_matrix: np.ndarray
    if auto_align_plane:
        rot_matrix, _, _, _ = compute_heat_repeats_alignment_matrix(
            kap_xyz=kap_xyz,
            heat5_range=heat5_range,
            heat6_range=heat6_range,
        )
    else:
        th_x: float = float(np.radians(rotation_x_deg))
        th_y: float = float(np.radians(rotation_y_deg))
        th_z: float = float(np.radians(rotation_z_deg))
        rx: np.ndarray = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(th_x), -np.sin(th_x)], [0.0, np.sin(th_x), np.cos(th_x)]])
        ry: np.ndarray = np.array([[np.cos(th_y), 0.0, np.sin(th_y)], [0.0, 1.0, 0.0], [-np.sin(th_y), 0.0, np.cos(th_y)]])
        rz: np.ndarray = np.array([[np.cos(th_z), -np.sin(th_z), 0.0], [np.sin(th_z), np.cos(th_z), 0.0], [0.0, 0.0, 1.0]])
        rot_matrix = np.dot(rz, np.dot(rx, ry))

    kap_center: np.ndarray = np.mean(kap_xyz, axis=0, keepdims=True)
    kap_xyz_rot: np.ndarray = np.dot(kap_xyz - kap_center, rot_matrix.T)
    state_xyz_rot: np.ndarray = np.dot(state_xyz - kap_center, rot_matrix.T)

    kap_2d: np.ndarray = kap_xyz_rot[:, :2]
    state_2d: np.ndarray = state_xyz_rot[:, :2].copy()

    bound_mask: np.ndarray = (
        np.ones(n_clusters, dtype=bool) if free_state_idx is None else np.arange(n_clusters) != free_state_idx
    )
    bound_states_2d: np.ndarray = state_2d[bound_mask]

    x_lim_min: float = 0.0
    x_lim_max: float = 0.0
    y_lim_min: float = 0.0
    y_lim_max: float = 0.0

    if zoom:
        x_min_b: float = float(np.min(bound_states_2d[:, 0]))
        x_max_b: float = float(np.max(bound_states_2d[:, 0]))
        y_min_b: float = float(np.min(bound_states_2d[:, 1]))
        y_max_b: float = float(np.max(bound_states_2d[:, 1]))

        c_x: float = (x_min_b + x_max_b) / 2.0
        c_y: float = (y_min_b + y_max_b) / 2.0
        c_offset: np.ndarray = np.array([c_x, c_y])
        kap_2d = kap_2d - c_offset
        state_2d = state_2d - c_offset

        target_span: float = 1.2
        span_x: float = max(x_max_b - x_min_b + 0.5, target_span)
        span_y: float = max(y_max_b - y_min_b + 0.5, target_span)
        zoom_span: float = max(span_x, span_y)
        half_span: float = zoom_span / 2.0

        x_lim_min = -half_span
        x_lim_max = half_span
        y_lim_min = -half_span
        y_lim_max = half_span

        if free_state_idx is not None:
            state_2d[free_state_idx] = np.array([x_lim_max - zoom_span * 0.15, y_lim_max - zoom_span * 0.15])
    else:
        x_min: float = float(np.min(kap_2d[:, 0]))
        x_max: float = float(np.max(kap_2d[:, 0]))
        y_min: float = float(np.min(kap_2d[:, 1]))
        y_max: float = float(np.max(kap_2d[:, 1]))
        c_x = (x_min + x_max) / 2.0
        c_y = (y_min + y_max) / 2.0
        c_offset = np.array([c_x, c_y])
        kap_2d = kap_2d - c_offset
        state_2d = state_2d - c_offset
        if free_state_idx is not None:
            state_2d[free_state_idx] = np.array([float(np.max(kap_2d[:, 0])) * 0.9, float(np.max(kap_2d[:, 1])) * 1.1])

    eigenvalues, eigenvectors = np.linalg.eig(msm_mat.T)
    eig_idx: int = int(np.argmin(np.abs(eigenvalues - 1.0)))
    stat_dist: np.ndarray = np.real(eigenvectors[:, eig_idx])
    stat_dist = stat_dist / np.sum(stat_dist)

    fig, ax = plt.subplots(figsize=(13.0, 9.5), dpi=dpi)

    res_indices: np.ndarray = np.arange(kap_n_ca, dtype=float)
    spline_2d: CubicSpline = CubicSpline(res_indices, kap_2d, bc_type="natural")

    # HEAT 5 and 6 ribbons
    dark_red_color: str = "#8B0000"
    light_red_color: str = "#FF6B6B"

    t_seg1: np.ndarray = np.linspace(heat5_range[0], min(heat5_range[1], kap_n_ca - 1), (heat5_range[1] - heat5_range[0]) * 20)
    seg1_coords: np.ndarray = spline_2d(t_seg1)
    ax.plot(
        seg1_coords[:, 0],
        seg1_coords[:, 1],
        color=dark_red_color,
        alpha=heat_ribbon_alpha,
        linewidth=heat_ribbon_width,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=3,
    )

    t_seg2: np.ndarray = np.linspace(heat6_range[0], min(heat6_range[1], kap_n_ca - 1), (heat6_range[1] - heat6_range[0]) * 20)
    seg2_coords: np.ndarray = spline_2d(t_seg2)
    ax.plot(
        seg2_coords[:, 0],
        seg2_coords[:, 1],
        color=light_red_color,
        alpha=heat_ribbon_alpha,
        linewidth=heat_ribbon_width,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=3,
    )

    rate_cmap: LinearSegmentedColormap = LinearSegmentedColormap.from_list("rate_bw", ["#D6D6D6", "#000000"])

    node_sizes: np.ndarray = (250.0 + 1400.0 * (stat_dist / np.max(stat_dist))) * state_size_factor
    node_radii: np.ndarray = np.sqrt(node_sizes) / 2.0

    positive_rates: np.ndarray = rate_mat[rate_mat > 0]
    top_threshold: float = float(np.percentile(positive_rates, top_rate_percentile)) if len(positive_rates) > 0 else 0.0
    max_rate: float = float(np.max(rate_mat)) if np.max(rate_mat) > 0 else 1.0

    # Draw static arrows
    arrow_patches: dict[tuple[int, int], FancyArrowPatch] = {}
    for i in range(n_clusters):
        for j in range(n_clusters):
            if i == j:
                continue
            rate: float = float(rate_mat[i, j])
            if rate >= top_threshold and rate > 0:
                norm_rate: float = rate / max_rate
                r_a: float = float(node_radii[i]) * 1.15
                r_b: float = float(node_radii[j]) * 1.15
                arr_c = "black" if arrow_color == "black" else rate_cmap(norm_rate)
                arr_patch: FancyArrowPatch = FancyArrowPatch(
                    posA=(state_2d[i, 0], state_2d[i, 1]),
                    posB=(state_2d[j, 0], state_2d[j, 1]),
                    connectionstyle="arc3,rad=0.18",
                    arrowstyle="-|>",
                    mutation_scale=16,
                    shrinkA=r_a,
                    shrinkB=r_b,
                    linewidth=1.0 + 3.0 * norm_rate,
                    color=arr_c,
                    alpha=0.85,
                    zorder=4,
                )
                ax.add_patch(arr_patch)
                arrow_patches[(i, j)] = arr_patch

    # Render static state nodes
    for i in range(n_clusters):
        unbound_frac: float = float(unbound_weights[i]) if (unbound_col_idx >= 0 and len(unbound_weights) > i) else 0.0
        interact_frac: float = max(0.0, min(1.0, 1.0 - unbound_frac))
        is_free: bool = (free_state_idx is not None) and (i == free_state_idx)

        # Uninteracting background circle
        ax.scatter(
            state_2d[i, 0],
            state_2d[i, 1],
            s=node_sizes[i],
            marker="o",
            facecolors="#F5F7FA",
            edgecolors="none",
            zorder=5.0,
        )

        # Interacting wedge
        if interact_frac > 0.001:
            theta_end: float = 90.0 + interact_frac * 360.0
            wedge_marker: mpath.Path = create_wedge_marker(theta1_deg=90.0, theta2_deg=theta_end)
            ax.scatter(
                state_2d[i, 0],
                state_2d[i, 1],
                s=node_sizes[i],
                marker=wedge_marker,
                facecolors="#4A6572",
                edgecolors="none",
                zorder=5.1,
            )

        # Outer border
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

        # Native-like accent ring
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

    heat_proxies: list[object] = [
        Line2D([0], [0], color=dark_red_color, linewidth=3.5, alpha=0.75, label=f"HEAT 5 ({heat5_range[0]}–{heat5_range[1]})"),
        Line2D([0], [0], color=light_red_color, linewidth=3.5, alpha=0.75, label=f"HEAT 6 ({heat6_range[0]}–{heat6_range[1]})"),
    ]
    state_frac_proxies: list[object] = [
        Line2D([0], [0], marker="o", color="w", label="Interacting fraction", markerfacecolor="#4A6572", markeredgecolor="#1A252C", markersize=18),
        Line2D([0], [0], marker="o", color="w", label="Non-interacting fraction", markerfacecolor="#F5F7FA", markeredgecolor="#1A252C", markersize=18),
    ]
    custom_proxies: list[object] = heat_proxies + state_frac_proxies
    if len(native_state_indices) > 0:
        native_label_str: str = f"Native-like ({', '.join(map(str, native_state_indices))})"
        custom_proxies.append(
            Line2D([0], [0], marker="o", color="w", label=native_label_str, markerfacecolor="none", markeredgecolor="#FF8C00", markeredgewidth=2.5, markersize=21)
        )
    ax.legend(handles=custom_proxies, loc="upper left", fontsize=19.6)

    if zoom:
        ax.set_xlim(x_lim_min, x_lim_max)
        ax.set_ylim(y_lim_min, y_lim_max)

    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, symmetric=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, symmetric=True))
    ax.set_xlabel(r"$x$ (nm)", fontsize=25.5)
    ax.set_ylabel(r"$y$ (nm)", fontsize=25.5)
    ax.tick_params(labelsize=23.25)

    # Dynamic elements for animation:
    # 1. Active state halo & core (matching Block 4 / Figure 2a styling)
    init_s_id: int = int(timeline[0]["state_id"]) if len(timeline) > 0 else 0
    init_pos_x: float = float(state_2d[init_s_id, 0])
    init_pos_y: float = float(state_2d[init_s_id, 1])
    init_node_size: float = float(node_sizes[init_s_id])

    active_halo = ax.scatter(
        [init_pos_x],
        [init_pos_y],
        s=[init_node_size * 1.55],
        marker="o",
        facecolors="#FF1744",
        edgecolors="#FF8A80",
        linewidths=2.2,
        alpha=0.45,
        zorder=10.0,
    )

    active_core = ax.scatter(
        [init_pos_x],
        [init_pos_y],
        s=[init_node_size * 0.35],
        marker="o",
        facecolors="#D50000",
        edgecolors="white",
        linewidths=1.8,
        alpha=1.0,
        zorder=10.1,
    )

    # 2. Active transition red arrow (matching Block 4 red #D50000)
    active_arrow: FancyArrowPatch = FancyArrowPatch(
        posA=(0.0, 0.0),
        posB=(0.0, 0.0),
        connectionstyle="arc3,rad=0.18",
        arrowstyle="-|>",
        mutation_scale=20,
        linewidth=3.8,
        color="#D50000",
        alpha=0.0,
        zorder=9,
    )
    ax.add_patch(active_arrow)

    # 3. Status text banner
    status_text = ax.text(
        0.5,
        1.03,
        "",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=20.0,
        fontweight="bold",
        color="#1A252C",
        path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
    )

    # 4. Progress bar at bottom
    progress_bar_bg = ax.plot([x_lim_min, x_lim_max], [y_lim_min + 0.02 * (y_lim_max - y_lim_min)] * 2, color="#E0E0E0", linewidth=6.0, zorder=9)[0]
    progress_bar_fg = ax.plot([x_lim_min, x_lim_min], [y_lim_min + 0.02 * (y_lim_max - y_lim_min)] * 2, color="#D50000", linewidth=6.0, zorder=10)[0]

    plt.tight_layout()

    # Animation update function
    def update_frame(frame_entry: StateTimelineEntry) -> list[object]:
        s_id: int = frame_entry["state_id"]
        from_s: int = frame_entry["transition_from"]
        to_s: int = frame_entry["transition_to"]
        is_trans: bool = frame_entry["is_transition"]
        trans_prog: float = frame_entry["transition_progress"]

        # Update position and size for active halo & core
        cur_x: float
        cur_y: float
        cur_size: float
        if is_trans and trans_prog > 0.0:
            pA: np.ndarray = state_2d[from_s]
            pB: np.ndarray = state_2d[to_s]
            dx: float = float(pB[0] - pA[0])
            dy: float = float(pB[1] - pA[1])
            m_pt: np.ndarray = (pA + pB) / 2.0
            p_ctrl: np.ndarray = np.array([m_pt[0] + 0.18 * dy, m_pt[1] - 0.18 * dx])
            t: float = trans_prog
            pos_t: np.ndarray = ((1.0 - t) ** 2) * pA + (2.0 * (1.0 - t) * t) * p_ctrl + (t ** 2) * pB
            cur_x = float(pos_t[0])
            cur_y = float(pos_t[1])
            cur_size = (1.0 - t) * float(node_sizes[from_s]) + t * float(node_sizes[to_s])
        else:
            cur_x = float(state_2d[s_id, 0])
            cur_y = float(state_2d[s_id, 1])
            cur_size = float(node_sizes[s_id])

        active_halo.set_offsets([[cur_x, cur_y]])
        active_halo.set_sizes([cur_size * 1.55])
        active_core.set_offsets([[cur_x, cur_y]])
        active_core.set_sizes([cur_size * 0.35])

        # Highlight transition arrow in red
        if is_trans:
            active_arrow.set_positions((state_2d[from_s, 0], state_2d[from_s, 1]), (state_2d[to_s, 0], state_2d[to_s, 1]))
            if (from_s, to_s) in arrow_patches:
                base_patch: FancyArrowPatch = arrow_patches[(from_s, to_s)]
                active_arrow.shrinkA = base_patch.shrinkA
                active_arrow.shrinkB = base_patch.shrinkB
            else:
                active_arrow.shrinkA = float(node_radii[from_s]) * 1.15
                active_arrow.shrinkB = float(node_radii[to_s]) * 1.15
            active_arrow.set_alpha(0.95)
        else:
            active_arrow.set_alpha(0.0)

        # Update text
        trans_str: str = f"  |  Transition: {from_s} \u2192 {to_s}" if is_trans else ""
        status_text.set_text(
            f"Time: {frame_entry['sim_time_ns']:.1f} ns ({frame_entry['sim_time_us']:.2f} \u00b5s)  |  Markov State: {s_id}{trans_str}"
        )

        # Update progress bar
        p_x: float = x_lim_min + frame_entry["progress_fraction"] * (x_lim_max - x_lim_min)
        progress_bar_fg.set_xdata([x_lim_min, p_x])

        return [active_halo, active_core, active_arrow, status_text, progress_bar_fg]

    os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)
    writer = anim.FFMpegWriter(
        fps=fps,
        codec="libx264",
        extra_args=["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-crf", "18", "-pix_fmt", "yuv420p"],
    )

    print(f">> Rendering iMSM animation video to: {output_video_path}")
    print(f"   Total video frames: {len(timeline)}, Framerate: {fps} fps")

    total_f: int = len(timeline)
    with writer.saving(fig, output_video_path, dpi=dpi):
        for idx, entry in enumerate(timeline):
            update_frame(entry)
            writer.grab_frame()
            if (idx + 1) % 10 == 0 or (idx + 1) == total_f:
                pct: float = ((idx + 1) / float(total_f)) * 100.0
                bar_len: int = 30
                filled: int = int(bar_len * (idx + 1) / float(total_f))
                bar: str = "=" * filled + "-" * (bar_len - filled)
                print(f"\r>> [Block 2] Rendering iMSM video: [{bar}] {pct:5.1f}% ({idx + 1}/{total_f} frames)", end="", flush=True)
    print()

    plt.close(fig)
    print(f">> Done rendering iMSM animation video: {output_video_path}")
    return output_video_path


def combine_simulation_and_imsm_videos(
    vmd_video_path: str,
    imsm_video_path: str,
    output_video_path: str,
    ffmpeg_path: str,
    target_height: int,
    fps: int,
) -> str:
    """Combine VMD 3D simulation video and 2D iMSM network video side-by-side into a single MP4."""
    if not os.path.exists(vmd_video_path):
        raise FileNotFoundError(f"VMD video not found: {vmd_video_path}. Render Block 1 first.")
    if not os.path.exists(imsm_video_path):
        raise FileNotFoundError(f"iMSM video not found: {imsm_video_path}. Render Block 2 first.")

    os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)

    filter_complex: str = (
        f"[0:v]fps={fps},scale=-2:{target_height}:flags=lanczos[v0];"
        f"[1:v]fps={fps},scale=-2:{target_height}:flags=lanczos[v1];"
        f"[v0][v1]hstack=inputs=2:shortest=1[v]"
    )

    cmd: list[str] = [
        ffmpeg_path,
        "-y",
        "-i",
        os.path.abspath(vmd_video_path),
        "-i",
        os.path.abspath(imsm_video_path),
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
        "-r",
        str(fps),
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        "-progress",
        "pipe:1",
        os.path.abspath(output_video_path),
    ]

    print(f">> Running ffmpeg to combine videos side-by-side into: {output_video_path}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    last_frame_reported: str = "0"
    if proc.stdout is not None:
        for line in proc.stdout:
            line_str: str = line.strip()
            if line_str.startswith("frame="):
                frame_val: str = line_str.split("=")[1].strip()
                last_frame_reported = frame_val
                print(f"\r>> [Block 3] Combining videos: encoded {frame_val} frames", end="", flush=True)
            elif line_str.startswith("progress=end"):
                print(f"\r>> [Block 3] Combining videos: encoded {last_frame_reported} frames (100.0%)", end="", flush=True)

    _, stderr_text = proc.communicate()
    print()

    if proc.returncode != 0:
        if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 1000:
            print(f">> Notice: ffmpeg exited with non-zero code {proc.returncode}, but output video was created successfully at: {output_video_path}")
            if stderr_text:
                last_lines = [l for l in stderr_text.strip().splitlines() if l][-2:]
                print(f"   ffmpeg info: {' | '.join(last_lines)}")
        else:
            raise RuntimeError(
                f"ffmpeg failed with exit code {proc.returncode}.\n"
                f"Command: {' '.join(cmd)}\n"
                f"Stderr: {stderr_text}"
            )

    print(f">> Successfully combined videos into: {output_video_path}")
    return output_video_path


def get_video_duration(video_path: str, ffprobe_path: str) -> float:
    """Probe and return the video duration in seconds using ffprobe."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not os.path.exists(ffprobe_path):
        raise FileNotFoundError(f"ffprobe binary not found at: {ffprobe_path}")

    cmd: list[str] = [
        ffprobe_path,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        os.path.abspath(video_path),
    ]
    result: subprocess.CompletedProcess[str] = subprocess.run(
        cmd, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffprobe failed to inspect {video_path} with code {result.returncode}: {result.stderr.strip()}"
        )
    duration_str: str = result.stdout.strip()
    if not duration_str:
        raise ValueError(f"ffprobe returned empty duration for {video_path}")
    return float(duration_str)


def combine_videos_with_fades(
    video_paths: list[str],
    output_video_path: str,
    transition_types: list[str],
    transition_duration_sec: float,
    target_width: int,
    target_height: int,
    fps: int,
    pad_colors: list[str],
    ffmpeg_path: str,
    ffprobe_path: str,
    crf: int,
    overlay_texts: list[str],
    text_font: str,
    text_fontsize: int,
    text_font_colors: list[str],
    text_border_colors: list[str],
    text_border_width: int,
    text_x: str,
    text_y: str,
    text_line_spacing: int,
    box: bool,
    box_colors: list[str],
    box_border_colors: list[str],
    box_border_width: int,
    box_padding: int,
) -> str:
    """Combine multiple videos into a single video with fade transitions and text overlays using ffmpeg."""
    n_videos: int = len(video_paths)
    if n_videos < 2:
        raise ValueError(f"At least two videos are required to combine with fades, got {n_videos}.")
    if len(transition_types) != n_videos - 1:
        raise ValueError(
            f"Expected {n_videos - 1} transition types for {n_videos} videos, got {len(transition_types)}."
        )
    if len(pad_colors) != n_videos:
        raise ValueError(
            f"Expected {n_videos} pad colors for {n_videos} videos, got {len(pad_colors)}."
        )
    if len(overlay_texts) != n_videos:
        raise ValueError(
            f"Expected {n_videos} overlay texts for {n_videos} videos, got {len(overlay_texts)}."
        )
    if len(text_font_colors) != n_videos:
        raise ValueError(
            f"Expected {n_videos} text font colors for {n_videos} videos, got {len(text_font_colors)}."
        )
    if len(text_border_colors) != n_videos:
        raise ValueError(
            f"Expected {n_videos} text border colors for {n_videos} videos, got {len(text_border_colors)}."
        )
    if box and len(box_colors) != n_videos:
        raise ValueError(
            f"Expected {n_videos} box colors for {n_videos} videos, got {len(box_colors)}."
        )
    if box and len(box_border_colors) != n_videos:
        raise ValueError(
            f"Expected {n_videos} box border colors for {n_videos} videos, got {len(box_border_colors)}."
        )
    for path in video_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Input video not found: {path}")
    if not os.path.exists(ffmpeg_path):
        raise FileNotFoundError(f"ffmpeg binary not found at: {ffmpeg_path}")
    if not os.path.exists(ffprobe_path):
        raise FileNotFoundError(f"ffprobe binary not found at: {ffprobe_path}")

    durations: list[float] = [
        get_video_duration(video_path=p, ffprobe_path=ffprobe_path) for p in video_paths
    ]
    for i, dur in enumerate(durations):
        if dur <= transition_duration_sec:
            raise ValueError(
                f"Video {video_paths[i]} duration ({dur:.2f}s) is shorter than transition duration ({transition_duration_sec:.2f}s)."
            )

    os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)

    text_files: list[str] = []
    for i, text in enumerate(overlay_texts):
        if text.strip():
            tf_path: str = os.path.join(
                os.path.dirname(os.path.abspath(output_video_path)),
                f".overlay_text_{i}.txt",
            )
            with open(tf_path, "w", encoding="utf-8") as f:
                f.write(text)
            text_files.append(tf_path)
        else:
            text_files.append("")

    filter_parts: list[str] = []
    for i in range(n_videos):
        base_filter: str = (
            f"[{i}:v]fps={fps},scale={target_width}:{target_height}:force_original_aspect_ratio=decrease:flags=lanczos,"
            f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2:color={pad_colors[i]},setsar=1,format=yuv420p"
        )
        if text_files[i]:
            escaped_tf: str = text_files[i].replace(":", "\\:")
            escaped_x: str = text_x.replace(",", "\\,")
            escaped_y: str = text_y.replace(",", "\\,")
            if box and box_border_width > 0:
                base_filter = (
                    f"{base_filter},drawtext=textfile='{escaped_tf}':font='{text_font}':"
                    f"fontsize={text_fontsize}:fontcolor=white@0:"
                    f"x={escaped_x}:y={escaped_y}:line_spacing={text_line_spacing}:"
                    f"box=1:boxcolor={box_border_colors[i]}:"
                    f"boxborderw={box_padding + box_border_width}"
                )
            box_opt: str = (
                f":box=1:boxcolor={box_colors[i]}:boxborderw={box_padding}"
                if box
                else ""
            )
            base_filter = (
                f"{base_filter},drawtext=textfile='{escaped_tf}':font='{text_font}':"
                f"fontsize={text_fontsize}:fontcolor={text_font_colors[i]}:"
                f"bordercolor={text_border_colors[i]}:borderw={text_border_width}:"
                f"x={escaped_x}:y={escaped_y}:line_spacing={text_line_spacing}"
                f"{box_opt}"
            )
        filter_parts.append(f"{base_filter}[v{i}]")

    current_acc_duration: float = durations[0]
    prev_stream: str = "v0"
    for j in range(n_videos - 1):
        next_stream: str = f"v{j + 1}"
        offset: float = current_acc_duration - transition_duration_sec
        current_acc_duration = current_acc_duration + durations[j + 1] - transition_duration_sec
        out_stream: str = "vout" if j == n_videos - 2 else f"x{j}"
        transition_name: str = transition_types[j]
        filter_parts.append(
            f"[{prev_stream}][{next_stream}]xfade=transition={transition_name}:"
            f"duration={transition_duration_sec:.3f}:offset={offset:.3f}[{out_stream}]"
        )
        prev_stream = out_stream

    filter_complex: str = ";".join(filter_parts)

    cmd: list[str] = [ffmpeg_path, "-y"]
    for path in video_paths:
        cmd.extend(["-i", os.path.abspath(path)])
    cmd.extend([
        "-filter_complex",
        filter_complex,
        "-map",
        "[vout]",
        "-r",
        str(fps),
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        "-progress",
        "pipe:1",
        os.path.abspath(output_video_path),
    ])

    print(f">> Running ffmpeg to combine {n_videos} videos with fades into: {output_video_path}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    last_frame_reported: str = "0"
    if proc.stdout is not None:
        for line in proc.stdout:
            line_str: str = line.strip()
            if line_str.startswith("frame="):
                frame_val: str = line_str.split("=")[1].strip()
                last_frame_reported = frame_val
                print(f"\r>> [Block 6] Combining videos with fades: encoded {frame_val} frames", end="", flush=True)
            elif line_str.startswith("progress=end"):
                print(f"\r>> [Block 6] Combining videos with fades: encoded {last_frame_reported} frames (100.0%)", end="", flush=True)

    _, stderr_text = proc.communicate()
    print()

    for tf in text_files:
        if tf and os.path.exists(tf):
            try:
                os.remove(tf)
            except OSError:
                pass

    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed with exit code {proc.returncode}.\n"
            f"Command: {' '.join(cmd)}\n"
            f"Stderr: {stderr_text}"
        )

    print(f">> Successfully combined videos into: {output_video_path}")
    return output_video_path



def compute_committor_transition_matrices(
    transition_matrix: np.ndarray,
    nuc_local_idx: int,
    cyt_local_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute forward and reverse Doob h-transformed transition matrices conditioned on translocation."""
    n_states: int = transition_matrix.shape[0]
    int_indices: list[int] = [i for i in range(n_states) if i not in (nuc_local_idx, cyt_local_idx)]

    # Forward committor q_fwd: q(nuc) = 0, q(cyt) = 1
    m_fwd: np.ndarray = np.eye(len(int_indices), dtype=float) - transition_matrix[np.ix_(int_indices, int_indices)]
    b_fwd: np.ndarray = transition_matrix[int_indices, cyt_local_idx]
    q_fwd_int: np.ndarray = np.linalg.lstsq(m_fwd, b_fwd, rcond=None)[0]
    q_fwd_int = np.clip(q_fwd_int, 0.0, 1.0)

    q_fwd: np.ndarray = np.zeros(n_states, dtype=float)
    q_fwd[cyt_local_idx] = 1.0
    q_fwd[int_indices] = q_fwd_int

    p_fwd: np.ndarray = np.zeros_like(transition_matrix, dtype=float)
    for i in range(n_states):
        weights_fwd: np.ndarray = transition_matrix[i, :] * q_fwd
        sum_fwd: float = float(np.sum(weights_fwd))
        if sum_fwd > 0.0:
            p_fwd[i, :] = weights_fwd / sum_fwd
        else:
            p_fwd[i, cyt_local_idx] = 1.0

    # Reverse committor q_rev: q(nuc) = 1, q(cyt) = 0
    m_rev: np.ndarray = np.eye(len(int_indices), dtype=float) - transition_matrix[np.ix_(int_indices, int_indices)]
    b_rev: np.ndarray = transition_matrix[int_indices, nuc_local_idx]
    q_rev_int: np.ndarray = np.linalg.lstsq(m_rev, b_rev, rcond=None)[0]
    q_rev_int = np.clip(q_rev_int, 0.0, 1.0)

    q_rev: np.ndarray = np.zeros(n_states, dtype=float)
    q_rev[nuc_local_idx] = 1.0
    q_rev[int_indices] = q_rev_int

    p_rev: np.ndarray = np.zeros_like(transition_matrix, dtype=float)
    for i in range(n_states):
        weights_rev: np.ndarray = transition_matrix[i, :] * q_rev
        sum_rev: float = float(np.sum(weights_rev))
        if sum_rev > 0.0:
            p_rev[i, :] = weights_rev / sum_rev
        else:
            p_rev[i, nuc_local_idx] = 1.0

    return p_fwd, p_rev


def simulate_npc_translocation_walks(
    p_fwd: np.ndarray,
    p_rev: np.ndarray,
    good_cluster_indices: list[int],
    mus: np.ndarray,
    nuc_local_idx: int,
    cyt_local_idx: int,
    n_walks: int,
    dt_us: float,
    random_seed: int,
    max_steps_per_walk: int,
) -> list[NPCTrajectoryStep]:
    """Simulate alternating forward and reverse translocation walks between Nucleus and Cytoplasm."""
    rng: np.random.Generator = np.random.default_rng(random_seed)
    n_states: int = len(good_cluster_indices)
    steps: list[NPCTrajectoryStep] = []
    global_step_counter: int = 0
    cumulative_time_us: float = 0.0

    for walk_idx in range(n_walks):
        is_fwd: bool = (walk_idx % 2 == 0)
        curr_local_idx: int = nuc_local_idx if is_fwd else cyt_local_idx
        target_local_idx: int = cyt_local_idx if is_fwd else nuc_local_idx
        trans_matrix: np.ndarray = p_fwd if is_fwd else p_rev
        direction_label: str = "Nucleus \u2192 Cytoplasm" if is_fwd else "Cytoplasm \u2192 Nucleus"

        step_in_walk: int = 0
        while curr_local_idx != target_local_idx and step_in_walk < max_steps_per_walk:
            probs: np.ndarray = trans_matrix[curr_local_idx, :]
            next_local_idx: int = int(rng.choice(n_states, p=probs))

            from_pos: tuple[float, float] = (float(mus[curr_local_idx][0]), float(mus[curr_local_idx][1]))
            to_pos: tuple[float, float] = (float(mus[next_local_idx][0]), float(mus[next_local_idx][1]))
            cumulative_time_us += dt_us

            step_entry: NPCTrajectoryStep = {
                "walk_index": walk_idx + 1,
                "step_in_walk": step_in_walk + 1,
                "global_step": global_step_counter + 1,
                "from_local_idx": curr_local_idx,
                "to_local_idx": next_local_idx,
                "from_cluster_id": good_cluster_indices[curr_local_idx],
                "to_cluster_id": good_cluster_indices[next_local_idx],
                "from_pos": from_pos,
                "to_pos": to_pos,
                "direction": direction_label,
                "sim_time_us": cumulative_time_us,
            }
            steps.append(step_entry)

            curr_local_idx = next_local_idx
            step_in_walk += 1
            global_step_counter += 1

    return steps


def build_npc_animation_frames(
    steps: list[NPCTrajectoryStep],
    target_duration_sec: float,
    fps: int,
    arrival_dwell_sec: float,
) -> list[NPCFrameState]:
    """Interpolate trajectory steps into video frame states with dwell and smooth glide phases."""
    if len(steps) == 0:
        return []

    target_total_frames: int = int(target_duration_sec * fps)
    walk_indices: list[int] = sorted(list({s["walk_index"] for s in steps}))
    n_walks: int = len(walk_indices)

    arrival_dwell_frames: int = max(5, int(arrival_dwell_sec * fps))
    total_arrival_frames: int = n_walks * arrival_dwell_frames

    remaining_frames: int = max(len(steps) * 6, target_total_frames - total_arrival_frames)
    frames_per_step: int = max(6, remaining_frames // len(steps))
    dwell_frames_per_step: int = max(2, int(0.35 * frames_per_step))
    glide_frames_per_step: int = frames_per_step - dwell_frames_per_step

    frames: list[NPCFrameState] = []
    current_frame_idx: int = 0

    first_step: NPCTrajectoryStep = steps[0]
    for _ in range(arrival_dwell_frames):
        frames.append(
            NPCFrameState(
                frame_idx=current_frame_idx,
                walk_index=first_step["walk_index"],
                direction=first_step["direction"],
                current_cluster_id=first_step["from_cluster_id"],
                sim_time_us=0.0,
                marker_x=first_step["from_pos"][0],
                marker_y=first_step["from_pos"][1],
                is_dwelling=True,
                progress_fraction=0.0,
                active_edge_x=[],
                active_edge_y=[],
            )
        )
        current_frame_idx += 1

    for step_idx, step in enumerate(steps):
        for _ in range(dwell_frames_per_step):
            frames.append(
                NPCFrameState(
                    frame_idx=current_frame_idx,
                    walk_index=step["walk_index"],
                    direction=step["direction"],
                    current_cluster_id=step["from_cluster_id"],
                    sim_time_us=step["sim_time_us"],
                    marker_x=step["from_pos"][0],
                    marker_y=step["from_pos"][1],
                    is_dwelling=True,
                    progress_fraction=0.0,
                    active_edge_x=[],
                    active_edge_y=[],
                )
            )
            current_frame_idx += 1

        x_from, y_from = step["from_pos"]
        x_to, y_to = step["to_pos"]

        for g_idx in range(glide_frames_per_step):
            alpha: float = float(g_idx + 1) / float(glide_frames_per_step)
            ease_t: float = 0.5 * (1.0 - np.cos(np.pi * alpha))
            cur_x: float = (1.0 - ease_t) * x_from + ease_t * x_to
            cur_y: float = (1.0 - ease_t) * y_from + ease_t * y_to

            frames.append(
                NPCFrameState(
                    frame_idx=current_frame_idx,
                    walk_index=step["walk_index"],
                    direction=step["direction"],
                    current_cluster_id=step["to_cluster_id"] if alpha >= 0.5 else step["from_cluster_id"],
                    sim_time_us=step["sim_time_us"],
                    marker_x=cur_x,
                    marker_y=cur_y,
                    is_dwelling=False,
                    progress_fraction=0.0,
                    active_edge_x=[x_from, x_to],
                    active_edge_y=[y_from, y_to],
                )
            )
            current_frame_idx += 1

        is_last_step_of_walk: bool = (
            step_idx == len(steps) - 1 or steps[step_idx + 1]["walk_index"] != step["walk_index"]
        )
        if is_last_step_of_walk:
            for _ in range(arrival_dwell_frames):
                frames.append(
                    NPCFrameState(
                        frame_idx=current_frame_idx,
                        walk_index=step["walk_index"],
                        direction=step["direction"],
                        current_cluster_id=step["to_cluster_id"],
                        sim_time_us=step["sim_time_us"],
                        marker_x=x_to,
                        marker_y=y_to,
                        is_dwelling=True,
                        progress_fraction=0.0,
                        active_edge_x=[],
                        active_edge_y=[],
                    )
                )
                current_frame_idx += 1

    total_f: int = len(frames)
    denom: float = float(max(1, total_f - 1))
    final_frames: list[NPCFrameState] = []
    for f in frames:
        f_copy: NPCFrameState = dict(f)
        f_copy["progress_fraction"] = float(f["frame_idx"]) / denom
        final_frames.append(f_copy)

    return final_frames


def generate_npc_translocation_video(
    base_tm_path: str,
    base_cluster_path: str,
    output_video_path: str,
    nup_colors_path: str,
    transition_rate_colorbar_path: str,
    radius: int,
    n_sites: int,
    n_samples: int,
    n_walks: int,
    target_duration_sec: float,
    fps: int,
    arrival_dwell_sec: float,
    dt_us: float,
    time_step_us: float,
    random_seed: int,
    max_steps_per_walk: int,
    dpi: int,
    min_rate: float,
    max_rate: float,
    pie_scaling: float,
    angle_threshold_degrees: float,
    angle_shift_degrees: float,
    add_spoke_markings: bool,
    add_circle_area_text: bool,
) -> str:
    """Generate animated MP4 of 10 translocation walks across the Figure 2a 4-site NPC network."""
    if not os.path.exists(nup_colors_path):
        raise FileNotFoundError(f"Nup colors image not found at: {nup_colors_path}")
    if not os.path.exists(transition_rate_colorbar_path):
        raise FileNotFoundError(f"Transition rate colorbar image not found at: {transition_rate_colorbar_path}")
    nup_colors_img: np.ndarray = mpimg.imread(nup_colors_path)
    cbar_img: np.ndarray = mpimg.imread(transition_rate_colorbar_path)

    np.random.seed(random_seed)

    tm_path: str = base_tm_path.replace("#r#", str(radius)).replace("#n#", str(n_sites))
    cluster_path: str = base_cluster_path.replace("#r#", str(radius)).replace("#n#", str(n_sites))

    clusters, coordinate_edges, p_mat, _ = load_cluster_data(tm_path=tm_path, cluster_path=cluster_path)
    all_indices: list[int] = list(range(len(clusters)))

    unprojected_mus: np.ndarray = estimate_unprojected_mus(n_samples, clusters, coordinate_edges, all_indices)
    good_mesostate_indices: list[int] = pick_good_clusters_by_mu_angle(
        unprojected_mus,
        clusters,
        angle_threshold_degrees=angle_threshold_degrees,
        angle_shift_degrees=angle_shift_degrees,
    )

    mus, _ = estimate_cluters_mu_cov(n_samples, clusters, coordinate_edges, good_mesostate_indices)
    mus = adjust_nucleus_cytoplasm_mus(clusters, p_mat, good_mesostate_indices, mus, pie_scaling=pie_scaling)

    nuc_local_matches: list[int] = [
        i for i, c_idx in enumerate(good_mesostate_indices) if calc_non_spoke_cluster_makeup(clusters[c_idx])[0] > 0.99
    ]
    cyt_local_matches: list[int] = [
        i for i, c_idx in enumerate(good_mesostate_indices) if calc_non_spoke_cluster_makeup(clusters[c_idx])[-1] > 0.99
    ]

    if len(nuc_local_matches) == 0 or len(cyt_local_matches) == 0:
        raise ValueError("Could not find Nucleus or Cytoplasm mesostates within visible cluster set.")

    nuc_local_idx: int = nuc_local_matches[0]
    cyt_local_idx: int = cyt_local_matches[0]

    q_mat: np.ndarray = infinitesimal_generator(p_mat, dt=time_step_us)

    n_good: int = len(good_mesostate_indices)
    edge_mask: np.ndarray = np.zeros((n_good, n_good), dtype=bool)
    for i in range(n_good):
        c_i: int = good_mesostate_indices[i]
        for j in range(n_good):
            if i == j:
                edge_mask[i, j] = True
                continue
            c_j: int = good_mesostate_indices[j]
            rate_ij: float = float(q_mat[c_i, c_j] + q_mat[c_j, c_i]) / 2.0
            if rate_ij > min_rate:
                edge_mask[i, j] = True

    p_sub: np.ndarray = p_mat[np.ix_(good_mesostate_indices, good_mesostate_indices)]
    p_sub_filtered: np.ndarray = p_sub * edge_mask
    for i_state in range(n_good):
        off_diag: float = float(p_sub_filtered[i_state, :].sum() - p_sub_filtered[i_state, i_state])
        if off_diag <= 1e-12:
            p_sub_filtered[i_state, :] = p_sub[i_state, :]

    row_sums: np.ndarray = p_sub_filtered.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    p_norm: np.ndarray = p_sub_filtered / row_sums

    p_fwd, p_rev = compute_committor_transition_matrices(
        transition_matrix=p_norm,
        nuc_local_idx=nuc_local_idx,
        cyt_local_idx=cyt_local_idx,
    )

    trajectory_steps: list[NPCTrajectoryStep] = simulate_npc_translocation_walks(
        p_fwd=p_fwd,
        p_rev=p_rev,
        good_cluster_indices=good_mesostate_indices,
        mus=mus,
        nuc_local_idx=nuc_local_idx,
        cyt_local_idx=cyt_local_idx,
        n_walks=n_walks,
        dt_us=dt_us,
        random_seed=random_seed,
        max_steps_per_walk=max_steps_per_walk,
    )

    animation_frames: list[NPCFrameState] = build_npc_animation_frames(
        steps=trajectory_steps,
        target_duration_sec=target_duration_sec,
        fps=fps,
        arrival_dwell_sec=arrival_dwell_sec,
    )

    fig_width: float = 19.2
    fig_height: float = 10.8
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)

    ax = fig.add_axes([0.37, 0.08, 0.32, 0.84])
    ax.set_xlim(-30.0, 30.0)
    ax.set_ylim(-46.0, 46.0)
    ax.set_aspect("equal", adjustable="box")
    hide_axii_and_show_scale_bar(ax, show_scale_bar=True, show_scale_text=False)

    visualize_arrows_between_mesostates(
        p_mat,
        fig,
        ax,
        good_mesostate_indices,
        mus,
        show_colorbar_title=False,
        in_out_flow=None,
        show_colorbar=False,
        min_rate=min_rate,
        max_rate=max_rate,
        time_step_us=time_step_us,
    )

    visualize_pie_mesostates(
        clusters,
        p_mat,
        ax,
        good_mesostate_indices,
        mus,
        PIE_COLORS,
        add_nucleus_cytoplasm_text=True,
        pie_scaling=pie_scaling,
        dots_only=False,
        nucleus_cytoplasm_fontsize=36,
    )

    add_npc_scaffold_picture(ax)
    ax.set_aspect("equal", adjustable="box")

    if add_spoke_markings:
        spoke_labels: list[str] = ["Spoke I", "Spoke II", "Spoke III", "Spoke IV"]
        spoke_xs: list[float] = [-16.5, -5.5, 5.5, 16.5]
        spoke_y: float = 29.2
        for sx, slabel in zip(spoke_xs, spoke_labels):
            ax.text(
                sx,
                spoke_y,
                slabel,
                fontsize=12.0,
                fontweight="bold",
                color="#37474F",
                ha="center",
                va="center",
                zorder=9999,
            )

    active_edge_line = ax.plot([], [], color="#D50000", linewidth=3.2, alpha=0.9, zorder=9999)[0]

    active_halo: Circle = Circle(
        (0.0, 0.0),
        radius=2.8,
        facecolor="#FF1744",
        edgecolor="#FF8A80",
        linewidth=2.2,
        alpha=0.45,
        zorder=10000,
    )
    active_core: Circle = Circle(
        (0.0, 0.0),
        radius=1.3,
        facecolor="#D50000",
        edgecolor="white",
        linewidth=1.8,
        zorder=10001,
    )
    ax.add_patch(active_halo)
    ax.add_patch(active_core)

    plt.draw()
    bbox = ax.get_position()

    # Left: vertical NupColors.png (centered vertically, anchored right against graph, half-size)
    nup_h: float = bbox.height * 0.425
    nup_bottom: float = bbox.y0 + (bbox.height - nup_h) / 2.0
    nup_w: float = (nup_h * fig_height * (418.0 / 934.0)) / fig_width
    margin_right: float = bbox.x0 - 0.025
    nup_left: float = margin_right - nup_w
    legend_ax = fig.add_axes([nup_left, nup_bottom, nup_w, nup_h])
    legend_ax.imshow(nup_colors_img)
    legend_ax.axis("off")

    if add_circle_area_text:
        stat_text_x: float = nup_left + nup_w / 2.0
        stat_text_y: float = nup_bottom + nup_h + 0.04
        fig.text(
            stat_text_x,
            stat_text_y,
            r"Circle Area $\propto$ Stationary",
            fontsize=13.0,
            fontweight="bold",
            color="#263238",
            ha="center",
            va="bottom",
        )

    # Right: vertical TransitionRateColorbar.png (centered vertically, close to content)
    cbar_h: float = bbox.height * 0.85
    cbar_bottom: float = bbox.y0 + (bbox.height - cbar_h) / 2.0
    cbar_w: float = (cbar_h * fig_height * (304.0 / 1234.0)) / fig_width
    cbar_left: float = bbox.x1 + 0.02
    cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_w, cbar_h])
    cbar_ax.imshow(cbar_img)
    cbar_ax.axis("off")

    prog_ax = fig.add_axes([bbox.x0, 0.035, bbox.width, 0.008])
    prog_ax.set_xlim(0.0, 1.0)
    prog_ax.set_ylim(0.0, 1.0)
    prog_ax.axis("off")
    prog_ax.plot([0.0, 1.0], [0.5, 0.5], color="#E0E0E0", linewidth=4.5, solid_capstyle="round")
    progress_bar_fg = prog_ax.plot([0.0, 0.0], [0.5, 0.5], color="#D50000", linewidth=4.5, solid_capstyle="round")[0]

    def update_frame(frame: NPCFrameState) -> list[object]:
        active_edge_line.set_data(frame["active_edge_x"], frame["active_edge_y"])
        active_halo.center = (frame["marker_x"], frame["marker_y"])
        active_core.center = (frame["marker_x"], frame["marker_y"])

        p_frac: float = float(frame["progress_fraction"])
        progress_bar_fg.set_xdata([0.0, p_frac])

        return [active_edge_line, active_halo, active_core, progress_bar_fg]

    os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)
    writer = anim.FFMpegWriter(
        fps=fps,
        codec="libx264",
        extra_args=["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-crf", "18", "-pix_fmt", "yuv420p"],
    )

    print(f">> Rendering NPC translocation animation video to: {output_video_path}")
    print(f"   Total frames: {len(animation_frames)}, Framerate: {fps} fps, Walks: {n_walks}")

    total_f_npc: int = len(animation_frames)
    with writer.saving(fig, output_video_path, dpi=dpi):
        for idx, frame_state in enumerate(animation_frames):
            update_frame(frame_state)
            writer.grab_frame()
            if (idx + 1) % 10 == 0 or (idx + 1) == total_f_npc:
                pct: float = ((idx + 1) / float(total_f_npc)) * 100.0
                bar_len: int = 30
                filled: int = int(bar_len * (idx + 1) / float(total_f_npc))
                bar: str = "=" * filled + "-" * (bar_len - filled)
                print(f"\r>> [Block 4] Rendering NPC Translocation: [{bar}] {pct:5.1f}% ({idx + 1}/{total_f_npc} frames)", end="", flush=True)
    print()

    plt.close(fig)
    print(f">> Done rendering NPC translocation video: {output_video_path}")
    return output_video_path


def simulate_continuous_npc_walks(
    p_fwd: np.ndarray,
    p_rev: np.ndarray,
    good_cluster_indices: list[int],
    mus: np.ndarray,
    nuc_local_idx: int,
    cyt_local_idx: int,
    total_target_frames: int,
    fps: int,
    arrival_dwell_sec: float,
    dt_us: float,
    random_seed: int,
    max_steps_per_walk: int,
    step_dwell_frames: int,
    step_glide_frames: int,
) -> list[NPCFrameState]:
    """Simulate continuous back-and-forth NPC translocation walks until reaching total_target_frames."""
    arrival_dwell_frames: int = max(5, int(arrival_dwell_sec * float(fps)))
    rng: np.random.Generator = np.random.default_rng(random_seed)
    n_states: int = len(good_cluster_indices)

    frames: list[NPCFrameState] = []
    curr_sim_time_us: float = 0.0
    walk_idx: int = 0

    init_pos_x: float = float(mus[nuc_local_idx][0])
    init_pos_y: float = float(mus[nuc_local_idx][1])
    init_cluster_id: int = good_cluster_indices[nuc_local_idx]

    # Initial arrival dwell at starting mesostate
    for _ in range(arrival_dwell_frames):
        if len(frames) >= total_target_frames:
            break
        entry: NPCFrameState = {
            "frame_idx": len(frames),
            "walk_index": 1,
            "direction": "Nucleus \u2192 Cytoplasm",
            "current_cluster_id": init_cluster_id,
            "sim_time_us": 0.0,
            "marker_x": init_pos_x,
            "marker_y": init_pos_y,
            "is_dwelling": True,
            "progress_fraction": 0.0,
            "active_edge_x": [],
            "active_edge_y": [],
        }
        frames.append(entry)

    while len(frames) < total_target_frames:
        walk_idx += 1
        is_fwd: bool = (walk_idx % 2 != 0)
        curr_local_idx: int = nuc_local_idx if is_fwd else cyt_local_idx
        target_local_idx: int = cyt_local_idx if is_fwd else nuc_local_idx
        trans_matrix: np.ndarray = p_fwd if is_fwd else p_rev
        direction_label: str = "Nucleus \u2192 Cytoplasm" if is_fwd else "Cytoplasm \u2192 Nucleus"

        step_in_walk: int = 0
        while curr_local_idx != target_local_idx and step_in_walk < max_steps_per_walk and len(frames) < total_target_frames:
            probs: np.ndarray = trans_matrix[curr_local_idx, :]
            next_local_idx: int = int(rng.choice(n_states, p=probs))

            x_from: float = float(mus[curr_local_idx][0])
            y_from: float = float(mus[curr_local_idx][1])
            x_to: float = float(mus[next_local_idx][0])
            y_to: float = float(mus[next_local_idx][1])
            from_cluster: int = good_cluster_indices[curr_local_idx]
            to_cluster: int = good_cluster_indices[next_local_idx]

            curr_sim_time_us += dt_us

            # Dwell at from_pos
            for _ in range(step_dwell_frames):
                if len(frames) >= total_target_frames:
                    break
                dwell_entry: NPCFrameState = {
                    "frame_idx": len(frames),
                    "walk_index": walk_idx,
                    "direction": direction_label,
                    "current_cluster_id": from_cluster,
                    "sim_time_us": curr_sim_time_us,
                    "marker_x": x_from,
                    "marker_y": y_from,
                    "is_dwelling": True,
                    "progress_fraction": 0.0,
                    "active_edge_x": [],
                    "active_edge_y": [],
                }
                frames.append(dwell_entry)

            # Smooth glide along edge
            for g_idx in range(step_glide_frames):
                if len(frames) >= total_target_frames:
                    break
                alpha: float = float(g_idx + 1) / float(step_glide_frames)
                ease_t: float = 0.5 * (1.0 - np.cos(np.pi * alpha))
                cur_x: float = (1.0 - ease_t) * x_from + ease_t * x_to
                cur_y: float = (1.0 - ease_t) * y_from + ease_t * y_to
                glide_entry: NPCFrameState = {
                    "frame_idx": len(frames),
                    "walk_index": walk_idx,
                    "direction": direction_label,
                    "current_cluster_id": to_cluster if alpha >= 0.5 else from_cluster,
                    "sim_time_us": curr_sim_time_us,
                    "marker_x": cur_x,
                    "marker_y": cur_y,
                    "is_dwelling": False,
                    "progress_fraction": 0.0,
                    "active_edge_x": [x_from, x_to],
                    "active_edge_y": [y_from, y_to],
                }
                frames.append(glide_entry)

            curr_local_idx = next_local_idx
            step_in_walk += 1

        # Arrival dwell at destination
        arr_x: float = float(mus[curr_local_idx][0])
        arr_y: float = float(mus[curr_local_idx][1])
        arr_cluster: int = good_cluster_indices[curr_local_idx]
        for _ in range(arrival_dwell_frames):
            if len(frames) >= total_target_frames:
                break
            arr_entry: NPCFrameState = {
                "frame_idx": len(frames),
                "walk_index": walk_idx,
                "direction": direction_label,
                "current_cluster_id": arr_cluster,
                "sim_time_us": curr_sim_time_us,
                "marker_x": arr_x,
                "marker_y": arr_y,
                "is_dwelling": True,
                "progress_fraction": 0.0,
                "active_edge_x": [],
                "active_edge_y": [],
            }
            frames.append(arr_entry)

    final_frames: list[NPCFrameState] = frames[:total_target_frames]
    denom: float = float(max(1, total_target_frames - 1))
    for f in final_frames:
        f["progress_fraction"] = float(f["frame_idx"]) / denom

    return final_frames


def generate_npc_grid_translocation_video(
    base_tm_path: str,
    base_cluster_path: str,
    output_video_path: str,
    nup_colors_path: str,
    transition_rate_colorbar_path: str,
    radii: list[int],
    n_sites: list[int],
    n_samples: int,
    target_duration_sec: float,
    fps: int,
    arrival_dwell_sec: float,
    step_dwell_frames: int,
    step_glide_frames: int,
    dt_us: float,
    time_step_us: float,
    random_seed: int,
    max_steps_per_walk: int,
    dpi: int,
    min_rate: float,
    max_rate: float,
    pie_scaling: float,
    angle_threshold_degrees: float,
    angle_shift_degrees: float,
    swap_axes: bool,
    title: str,
    add_nucleus_cytoplasm_text: bool,
    nucleus_cytoplasm_top_row_only: bool,
    nucleus_cytoplasm_fontsize: float,
    add_circle_area_text: bool,
) -> str:
    """Generate animated MP4 of multiple NPC iMSM translocation networks running simultaneously on a 2D parameter grid."""
    if not os.path.exists(nup_colors_path):
        raise FileNotFoundError(f"Nup colors image not found at: {nup_colors_path}")
    if not os.path.exists(transition_rate_colorbar_path):
        raise FileNotFoundError(f"Transition rate colorbar image not found at: {transition_rate_colorbar_path}")
    nup_colors_img: np.ndarray = mpimg.imread(nup_colors_path)
    cbar_img: np.ndarray = mpimg.imread(transition_rate_colorbar_path)

    fig_cols: int = len(radii) if swap_axes else len(n_sites)
    fig_rows: int = len(n_sites) if swap_axes else len(radii)

    fig_width: float = 19.2
    fig_height: float = 10.8
    total_frames: int = int(target_duration_sec * float(fps))

    print(f">> Initializing NPC {fig_rows}x{fig_cols} grid translocation video (16:9)...")
    print(f"   Duration: {target_duration_sec:.1f}s ({total_frames} frames @ {fps} fps)")
    print(f"   Radii: {radii}, N_sites: {n_sites}, swap_axes: {swap_axes}")

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    if title:
        fig.suptitle(title, fontsize=18.0, fontweight="bold", y=0.96)

    grid_left: float = 0.35
    grid_right: float = 0.69
    grid_bottom: float = 0.08
    grid_top: float = 0.88

    gs = fig.add_gridspec(
        fig_rows,
        fig_cols,
        left=grid_left,
        right=grid_right,
        bottom=grid_bottom,
        top=grid_top,
        wspace=0.04,
        hspace=0.06,
    )
    axes: list[list[object]] = [
        [fig.add_subplot(gs[r_i, c_i]) for c_i in range(fig_cols)] for r_i in range(fig_rows)
    ]

    all_timelines: list[list[NPCFrameState]] = []
    active_edge_lines: list[Line2D] = []
    active_halos: list[Circle] = []
    active_cores: list[Circle] = []

    for j_r, r in enumerate(radii):
        for i_n, n in enumerate(n_sites):
            row_idx: int = i_n if swap_axes else j_r
            col_idx: int = j_r if swap_axes else i_n
            ax = axes[row_idx][col_idx]

            tm_path: str = base_tm_path.replace("#r#", str(r)).replace("#n#", str(n))
            cluster_path: str = base_cluster_path.replace("#r#", str(r)).replace("#n#", str(n))

            clusters, coordinate_edges, p_mat, _ = load_cluster_data(tm_path=tm_path, cluster_path=cluster_path)
            all_indices: list[int] = list(range(len(clusters)))

            unprojected_mus: np.ndarray = estimate_unprojected_mus(n_samples, clusters, coordinate_edges, all_indices)
            good_mesostate_indices: list[int] = pick_good_clusters_by_mu_angle(
                unprojected_mus,
                clusters,
                angle_threshold_degrees=angle_threshold_degrees,
                angle_shift_degrees=angle_shift_degrees,
            )

            mus, _ = estimate_cluters_mu_cov(n_samples, clusters, coordinate_edges, good_mesostate_indices)
            mus = adjust_nucleus_cytoplasm_mus(clusters, p_mat, good_mesostate_indices, mus, pie_scaling=pie_scaling)

            nuc_local_matches: list[int] = [
                i for i, c_idx in enumerate(good_mesostate_indices) if calc_non_spoke_cluster_makeup(clusters[c_idx])[0] > 0.99
            ]
            cyt_local_matches: list[int] = [
                i for i, c_idx in enumerate(good_mesostate_indices) if calc_non_spoke_cluster_makeup(clusters[c_idx])[-1] > 0.99
            ]

            if len(nuc_local_matches) == 0 or len(cyt_local_matches) == 0:
                raise ValueError(f"Could not find Nucleus or Cytoplasm mesostates for r={r}, n={n}.")

            nuc_local_idx: int = nuc_local_matches[0]
            cyt_local_idx: int = cyt_local_matches[0]

            q_mat: np.ndarray = infinitesimal_generator(p_mat, dt=time_step_us)

            n_good: int = len(good_mesostate_indices)
            edge_mask: np.ndarray = np.zeros((n_good, n_good), dtype=bool)
            for i in range(n_good):
                c_i: int = good_mesostate_indices[i]
                for j in range(n_good):
                    if i == j:
                        edge_mask[i, j] = True
                        continue
                    c_j: int = good_mesostate_indices[j]
                    rate_ij: float = float(q_mat[c_i, c_j] + q_mat[c_j, c_i]) / 2.0
                    if rate_ij > min_rate:
                        edge_mask[i, j] = True

            p_sub: np.ndarray = p_mat[np.ix_(good_mesostate_indices, good_mesostate_indices)]
            p_sub_filtered: np.ndarray = p_sub * edge_mask
            for i_state in range(n_good):
                off_diag: float = float(p_sub_filtered[i_state, :].sum() - p_sub_filtered[i_state, i_state])
                if off_diag <= 1e-12:
                    p_sub_filtered[i_state, :] = p_sub[i_state, :]

            row_sums: np.ndarray = p_sub_filtered.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0.0] = 1.0
            p_norm: np.ndarray = p_sub_filtered / row_sums

            p_fwd, p_rev = compute_committor_transition_matrices(
                transition_matrix=p_norm,
                nuc_local_idx=nuc_local_idx,
                cyt_local_idx=cyt_local_idx,
            )

            cell_seed: int = random_seed + row_idx * 100 + col_idx * 13
            timeline: list[NPCFrameState] = simulate_continuous_npc_walks(
                p_fwd=p_fwd,
                p_rev=p_rev,
                good_cluster_indices=good_mesostate_indices,
                mus=mus,
                nuc_local_idx=nuc_local_idx,
                cyt_local_idx=cyt_local_idx,
                total_target_frames=total_frames,
                fps=fps,
                arrival_dwell_sec=arrival_dwell_sec,
                dt_us=dt_us,
                random_seed=cell_seed,
                max_steps_per_walk=max_steps_per_walk,
                step_dwell_frames=step_dwell_frames,
                step_glide_frames=step_glide_frames,
            )
            all_timelines.append(timeline)

            ax.set_xlim(-30.0, 30.0)
            ax.set_ylim(-46.0, 46.0)
            ax.set_aspect("equal", adjustable="box")
            hide_axii_and_show_scale_bar(ax, show_scale_bar=False, show_scale_text=False)

            show_colorbar: bool = False
            visualize_arrows_between_mesostates(
                p_mat,
                fig,
                ax,
                good_mesostate_indices,
                mus,
                show_colorbar_title=False,
                in_out_flow=None,
                show_colorbar=show_colorbar,
                min_rate=min_rate,
                max_rate=max_rate,
                time_step_us=time_step_us,
            )

            show_nc_text: bool = add_nucleus_cytoplasm_text and (not nucleus_cytoplasm_top_row_only or row_idx == 0)
            visualize_pie_mesostates(
                clusters,
                p_mat,
                ax,
                good_mesostate_indices,
                mus,
                PIE_COLORS,
                add_nucleus_cytoplasm_text=show_nc_text,
                pie_scaling=pie_scaling,
                dots_only=False,
                nucleus_cytoplasm_fontsize=nucleus_cytoplasm_fontsize,
            )

            add_npc_scaffold_picture(ax)
            ax.set_aspect("equal", adjustable="box")

            if row_idx == 0:
                kda_val: float = float(radius_a_to_kda(r))
                ax.set_title(f"{kda_val:.0f} kDa", fontsize=14.0, fontweight="bold", pad=8.0, color="#1A237E")

            if col_idx == 0:
                ax.set_ylabel(f"{n} Sites", fontsize=15.0, fontweight="bold", labelpad=10.0, color="#1A237E")

            edge_line = ax.plot([], [], color="#D50000", linewidth=2.5, alpha=0.9, zorder=9999)[0]
            halo = Circle(
                (0.0, 0.0),
                radius=2.6,
                facecolor="#FF1744",
                edgecolor="#FF8A80",
                linewidth=1.8,
                alpha=0.45,
                zorder=10000,
            )
            core = Circle(
                (0.0, 0.0),
                radius=1.2,
                facecolor="#D50000",
                edgecolor="white",
                linewidth=1.5,
                zorder=10001,
            )
            ax.add_patch(halo)
            ax.add_patch(core)

            active_edge_lines.append(edge_line)
            active_halos.append(halo)
            active_cores.append(core)

    plt.draw()
    top_left_bbox = axes[0][0].get_position()
    top_right_bbox = axes[0][fig_cols - 1].get_position()
    bottom_right_bbox = axes[fig_rows - 1][fig_cols - 1].get_position()
    bottom_left_bbox = axes[fig_rows - 1][0].get_position()

    grid_h: float = top_right_bbox.y1 - bottom_right_bbox.y0

    # Left: vertical NupColors.png (centered vertically, anchored right against grid y-labels, half-size)
    nup_h: float = grid_h * 0.425
    nup_bottom: float = bottom_left_bbox.y0 + (grid_h - nup_h) / 2.0
    nup_w: float = (nup_h * fig_height * (418.0 / 934.0)) / fig_width
    nup_right: float = top_left_bbox.x0 - 0.045
    nup_left: float = nup_right - nup_w
    legend_ax = fig.add_axes([nup_left, nup_bottom, nup_w, nup_h])
    legend_ax.imshow(nup_colors_img)
    legend_ax.axis("off")

    if add_circle_area_text:
        stat_text_x: float = nup_left + nup_w / 2.0
        stat_text_y: float = nup_bottom + nup_h + 0.04
        fig.text(
            stat_text_x,
            stat_text_y,
            r"Circle Area $\propto$ Stationary",
            fontsize=13.0,
            fontweight="bold",
            color="#263238",
            ha="center",
            va="bottom",
        )

    # Right: vertical TransitionRateColorbar.png (centered vertically, close to content)
    cbar_h: float = grid_h * 0.85
    cbar_bottom: float = bottom_right_bbox.y0 + (grid_h - cbar_h) / 2.0
    cbar_w: float = (cbar_h * fig_height * (304.0 / 1234.0)) / fig_width
    cbar_left: float = top_right_bbox.x1 + 0.02
    cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_w, cbar_h])
    cbar_ax.imshow(cbar_img)
    cbar_ax.axis("off")

    prog_left: float = bottom_left_bbox.x0
    prog_w: float = bottom_right_bbox.x1 - bottom_left_bbox.x0
    prog_ax = fig.add_axes([prog_left, 0.025, prog_w, 0.008])
    prog_ax.set_xlim(0.0, 1.0)
    prog_ax.set_ylim(0.0, 1.0)
    prog_ax.axis("off")
    prog_ax.plot([0.0, 1.0], [0.5, 0.5], color="#E0E0E0", linewidth=4.5, solid_capstyle="round")
    progress_bar_fg = prog_ax.plot([0.0, 0.0], [0.5, 0.5], color="#D50000", linewidth=4.5, solid_capstyle="round")[0]

    n_cells: int = len(all_timelines)

    def update_frame(f_idx: int) -> list[object]:
        artists: list[object] = []
        for k in range(n_cells):
            state: NPCFrameState = all_timelines[k][f_idx]
            active_edge_lines[k].set_data(state["active_edge_x"], state["active_edge_y"])
            active_halos[k].center = (state["marker_x"], state["marker_y"])
            active_cores[k].center = (state["marker_x"], state["marker_y"])
            artists.extend([active_edge_lines[k], active_halos[k], active_cores[k]])

        p_frac: float = float(f_idx) / float(max(1, total_frames - 1))
        progress_bar_fg.set_xdata([0.0, p_frac])
        artists.append(progress_bar_fg)
        return artists

    os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)
    writer = anim.FFMpegWriter(
        fps=fps,
        codec="libx264",
        extra_args=["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-crf", "18", "-pix_fmt", "yuv420p"],
    )

    print(f">> Rendering NPC grid translocation video to: {output_video_path}")
    print(f"   Total frames: {total_frames}, Framerate: {fps} fps, Subplots: {n_cells}")

    with writer.saving(fig, output_video_path, dpi=dpi):
        for idx in range(total_frames):
            update_frame(idx)
            writer.grab_frame()
            if (idx + 1) % 10 == 0 or (idx + 1) == total_frames:
                pct: float = ((idx + 1) / float(total_frames)) * 100.0
                bar_len: int = 30
                filled: int = int(bar_len * (idx + 1) / float(total_frames))
                bar: str = "=" * filled + "-" * (bar_len - filled)
                print(f"\r>> [Block 5] Rendering NPC Grid Translocation: [{bar}] {pct:5.1f}% ({idx + 1}/{total_frames} frames)", end="", flush=True)
    print()

    plt.close(fig)
    print(f">> Done rendering NPC grid translocation video: {output_video_path}")
    return output_video_path


