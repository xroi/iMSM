# %% imports
import os
import shutil
import sys
from pathlib import Path

_fig_dir: Path = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
_repo_root: Path = _fig_dir.parent if _fig_dir.name == "figure_scripts" else _fig_dir
for _p in [str(_fig_dir), str(_repo_root)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

if not os.path.exists("data") and (_repo_root / "data").exists():
    os.chdir(_repo_root)

import importlib

import figure_videos_helpers
importlib.reload(figure_videos_helpers)
from figure_videos_helpers import (
    generate_vmd_movie_script,
    generate_imsm_network_video,
    combine_simulation_and_imsm_videos,
    generate_npc_translocation_video,
    generate_npc_grid_translocation_video,
    combine_videos_with_fades,
)


# %% Block 1: Generate VMD simulation movie script

save_dir_b1: str = "plots/videos"
vmd_movie_path_b1: str = os.path.join(save_dir_b1, "vmd_simulation.mp4")
vmd_script_dir_b1: str = "vmd_states/fgC88_91_com"

vmd_script_path: str = generate_vmd_movie_script(
    top_path="data/nup_sims/fsfgx2/output_from_0_nowat.dms.pdb",
    traj_paths=[
        "data/nup_sims/fsfgx2/output_from_0_to_2499_nowat_unwarped.dcd",
        "data/nup_sims/fsfgx2/output_from_2500_to_4999_nowat_unwarped.dcd",
        "data/nup_sims/fsfgx2/output_from_5000_to_7499_nowat_unwarped.dcd",
        "data/nup_sims/fsfgx2/output_from_7500_to_9999_nowat_unwarped.dcd",
        "data/nup_sims/fsfgx2/output_from_10000_to_12499_nowat_unwarped.dcd",
        "data/nup_sims/fsfgx2/output_from_12500_nowat_unwarped.dcd",
    ],
    imsm_checkpoint_path="data/nup_sims/fsfgx2/imsm/fgC88_91_com",
    output_dir=vmd_script_dir_b1,
    output_movie_path=vmd_movie_path_b1,
    kap_n_ca=861,
    focal_fg_alphacarbon=(88, 91),
    window_size=500,
    stride=10,
    fps=30,
    ns_per_frame=0.96,
    smoothing_window_ns=50.0,
    auto_align_plane=True,
    rotation_x_deg=0.0,
    rotation_y_deg=0.0,
    rotation_z_deg=0.0,
    heat5_range=(175, 195),
    heat6_range=(218, 232),
    vmd_zoom_scale=8.0,
    all_states_translate=(-0.3, 0.3, 0.0),
    width=1920,
    height=1080,
)

print(f">> Block 1 Complete: Generated VMD movie script at:\n   {vmd_script_path}")
print("   To render the VMD simulation video, execute in terminal:")
print(f"   vmd -e {vmd_script_path} -dispdev text")
print(f"   or in GUI VMD run: render_simulation_movie \"{vmd_movie_path_b1}\"")


# %% Block 2: Generate iMSM network animation video

save_dir_b2: str = "plots/videos"
imsm_movie_path_b2: str = os.path.join(save_dir_b2, "imsm_network.mp4")
top_path_b2: str = "data/nup_sims/fsfgx2/output_from_0_nowat.dms.pdb"
imsm_checkpoint_path_b2: str = "data/nup_sims/fsfgx2/imsm/fgC88_91_com"

imsm_video_result: str = generate_imsm_network_video(
    top_path=top_path_b2,
    imsm_checkpoint_path=imsm_checkpoint_path_b2,
    output_video_path=imsm_movie_path_b2,
    total_frames=14820,
    stride=10,
    window_size=500,
    fps=30,
    top_rate_percentile=10.0,
    kap_n_ca=861,
    repeat_colors=("#78909C", "#CFD8DC"),
    zoom=True,
    state_size_factor=5.0,
    free_threshold=0.9,
    dt_ns=1000.0,
    auto_align_plane=True,
    rotation_x_deg=0.0,
    rotation_y_deg=0.0,
    rotation_z_deg=0.0,
    heat5_range=(175, 195),
    heat6_range=(218, 232),
    focal_fg_alphacarbon=(88, 91),
    native_pdb_path=top_path_b2,
    interaction_capacity=5,
    max_surface_dist=1.0,
    native_similarity_threshold=0.75,
    top_n_print=10,
    heat_ribbon_width=30.0,
    heat_ribbon_alpha=0.25,
    state_color_mode="interaction_fraction",
    arrow_color=None,
    show_rate_colorbar=True,
    show_residue_colorbar=False,
    diverging_palette=(),
    residue_color_range=(1, 21),
    ns_per_frame=0.96,
    dpi=100,
    transition_glide_frames=12,
)

print(f">> Block 2 Complete: Generated iMSM animation video at:\n   {imsm_video_result}")


# %% Block 3: Combine simulation and iMSM videos side-by-side

save_dir_b3: str = "plots/videos"
vmd_movie_path_b3: str = os.path.join(save_dir_b3, "vmd_simulation.mp4")
imsm_movie_path_b3: str = os.path.join(save_dir_b3, "imsm_network.mp4")
combined_movie_path_b3: str = os.path.join(save_dir_b3, "figure_6_combined_video.mp4")

combined_video_result: str = combine_simulation_and_imsm_videos(
    vmd_video_path=vmd_movie_path_b3,
    imsm_video_path=imsm_movie_path_b3,
    output_video_path=combined_movie_path_b3,
    ffmpeg_path=shutil.which("ffmpeg") or "ffmpeg",
    target_height=1080,
    fps=30,
)

print(f">> Block 3 Complete: Generated combined synchronized video at:\n   {combined_video_result}")


# %% Block 4: Generate NPC 4-site translocation animation video (Figure 2a)

save_dir_b4: str = "plots/videos"
npc_movie_path_b4: str = os.path.join(save_dir_b4, "npc_4sites_imsm_trajectory.mp4")

npc_video_result: str = generate_npc_translocation_video(
    base_tm_path="data/ntr_variants/#n#_#r#_more/6_transition_matrices_subsets/1.00fraction_simulations/0index/320clusters.pickle",
    base_cluster_path="data/ntr_variants/#n#_#r#_more/5_clustering_subsets/1.00fraction_simulations/0index/320clusters.pickle",
    output_video_path=npc_movie_path_b4,
    nup_colors_path="plots/videos/NupColors.png",
    transition_rate_colorbar_path="plots/videos/TransitionRateColorbar.png",
    radius=26,
    n_sites=4,
    n_samples=2000,
    n_walks=5,
    target_duration_sec=15.0,
    fps=10,
    arrival_dwell_sec=0.5,
    dt_us=10.0,
    time_step_us=5.0,
    random_seed=42,
    max_steps_per_walk=150,
    dpi=100,
    min_rate=0.01,
    max_rate=5.0,
    pie_scaling=15.0,
    angle_threshold_degrees=92.0,
    angle_shift_degrees=0.0,
    add_spoke_markings=True,
    add_circle_area_text=True,
)

print(f">> Block 4 Complete: Generated NPC 4-site translocation video at:\n   {npc_video_result}")


# %% Block 5: Generate NPC 3x3 grid translocation animation video (Figure 3a)

save_dir_b5: str = "plots/videos"
npc_grid_movie_path_b5: str = os.path.join(save_dir_b5, "npc_grid_3x3_imsm_trajectory.mp4")

npc_grid_video_result: str = generate_npc_grid_translocation_video(
    base_tm_path="data/ntr_variants/#n#_#r#_more/6_transition_matrices_subsets/1.00fraction_simulations/0index/320clusters.pickle",
    base_cluster_path="data/ntr_variants/#n#_#r#_more/5_clustering_subsets/1.00fraction_simulations/0index/320clusters.pickle",
    output_video_path=npc_grid_movie_path_b5,
    nup_colors_path="plots/videos/NupColors.png",
    transition_rate_colorbar_path="plots/videos/TransitionRateColorbar.png",
    radii=[10, 18, 26],
    n_sites=[2, 4, 6],
    n_samples=2000,
    target_duration_sec=15.0,
    fps=10,
    arrival_dwell_sec=0.5,
    step_dwell_frames=3,
    step_glide_frames=7,
    dt_us=10.0,
    time_step_us=5.0,
    random_seed=43,
    max_steps_per_walk=150,
    dpi=100,
    min_rate=0.01,
    max_rate=10.0,
    pie_scaling=15.0,
    angle_threshold_degrees=92.0,
    angle_shift_degrees=0.0,
    swap_axes=True,
    title="",
    add_nucleus_cytoplasm_text=True,
    nucleus_cytoplasm_top_row_only=True,
    nucleus_cytoplasm_fontsize=11.0,
    add_circle_area_text=True,
)

print(f">> Block 5 Complete: Generated NPC 3x3 grid translocation video at:\n   {npc_grid_video_result}")


# %% Block 6: Combine simulation and translocation showcase videos with fades

save_dir_b6: str = "plots/videos"
npcsim_movie_path_b6: str = os.path.join(save_dir_b6, "npcsim-highres.mov")
npc_4sites_movie_path_b6: str = os.path.join(save_dir_b6, "npc_4sites_imsm_trajectory.mp4")
npc_grid_movie_path_b6: str = os.path.join(save_dir_b6, "npc_grid_3x3_imsm_trajectory.mp4")
combined_showcase_movie_path_b6: str = os.path.join(save_dir_b6, "npc_showcase_combined_fades.mp4")

overlay_texts_b6: list[str] = [
    "Input:\nBrownian dynamics\ntrajectories from integrative model\nof nucleocytoplasmic transport\n(Raveh, Eliasian et al., PNAS 2025)",
    "Output:\ninteraction-based Markov\nstate model of\nnucleocytoplasmic transport",
    "Output:\ninteraction-based Markov\nstate model of\nnucleocytoplasmic transport"
]

combined_showcase_result: str = combine_videos_with_fades(
    video_paths=[
        npcsim_movie_path_b6,
        npc_4sites_movie_path_b6,
        npc_grid_movie_path_b6,
    ],
    output_video_path=combined_showcase_movie_path_b6,
    transition_types=["fade", "fade"],
    transition_duration_sec=1.5,
    target_width=1920,
    target_height=1080,
    fps=30,
    pad_colors=["black", "white", "white"],
    ffmpeg_path=shutil.which("ffmpeg") or "ffmpeg",
    ffprobe_path=shutil.which("ffprobe") or "ffprobe",
    crf=18,
    overlay_texts=overlay_texts_b6,
    text_font="DejaVu Sans",
    text_fontsize=21,
    text_font_colors=["white", "black", "black"],
    text_border_colors=["black", "white", "white"],
    text_border_width=1,
    text_x="40",
    text_y="40",
    text_line_spacing=8,
    box=True,
    box_colors=["black@0.75", "white@0.9", "white@0.9"],
    box_border_colors=["gray@0.8", "gray@0.8", "gray@0.8"],
    box_border_width=2,
    box_padding=12,
)

print(f">> Block 6 Complete: Generated combined showcase video at:\n   {combined_showcase_result}")

