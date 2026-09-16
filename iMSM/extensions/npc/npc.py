import pickle
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional

# Path formatting helpers
def _cp(params: dict, *parts: object) -> Path:
    """Checkpoint path join helper (keeps stage code readable)."""
    return Path(params["CHECKPOINTS_PATH"]).joinpath(*(str(p) for p in parts))


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _time_range_strings(params: dict) -> tuple[str, str, str]:
    t0_str = f"{int(params['LOAD_MD_START_TIME_NS'] / 1000)}"
    t1_str = f"{int(params['LOAD_MD_END_TIME_NS'] / 1000)}"
    return t0_str, t1_str, f"{t0_str}-{t1_str}"


def _subset_folder(params: dict) -> str:
    mode_string = "_simulations" if params['DATA_SUBSET_MODE'] == "simulation" else ""
    return f"{params['DATA_SUBSET']:.2f}fraction{mode_string}/{params['DATA_SUBSET_INDEX']}index"


def _get_sim_indexes(params: dict) -> List[int]:
    sims = list(params.get('LOAD_MD_SIMS_RANGE', range(1, 31)))
    ignored = params.get('IGNORED_SIMULATIONS', [])
    return [s for s in sims if s not in ignored]

###

@dataclass
class iMSMConfig:
    # --- General ---
    checkpoints_path: str
    mode: str = "normal" #options: "normal", "bootstrap", "subset"
    # normal: normal run
    # perform bootstrap resampling to generate transition matrices and to computer permeabilities
    # subset: use only a subset of data for clustering and MSM generation
    
    # --- Bootstrapping ---
    bootstrap_repeats: int = 0 # if n>0 and bootstrap_mode is True, will do bootstrap resampling to generate n matrices (using the same clustering)
    
    # --- Data Subsetting ---
    # If data_subset <1.0, use a data subset of that fractionality for the clustering stage onwards.
    # Will equally collect data from multiple simulations.
    # For example, if there are 30 simulations of length 100us, and data_subset is 0.1, and data_subset_mode is "time", will keep the data_subset_index'th 10us of each simulation.
    # if data_subset_index is None, will take the final 10us.
    # if data_subset_mode is "simulation", will take the data_subset_index'th 10% of simulations fully.
    data_subset: float = 1.0
    data_subset_index: int = -1
    data_subset_mode: str = "time" # options: "time", "simulation"
    
    # --- Stage 1 & 2: Loading ---
    load_md_kap_sites: int = 2
    load_md_kap_radius: int = 20
    load_md_kap_amount: int = 50
    load_md_base_path: str = ""
    load_md_sims_range: range = range(1, 31)
    load_md_start_time_ns: int = 10000
    load_md_end_time_ns: int = 30000
    load_md_step_ns: int = 100
    load_md_ignored_nup_types: List[str] = field(default_factory=list)
    
    # --- Stage 3: Categorization ---
    interaction_capacity: int = 5
    max_surface_dist_nm: float = 1.0
    split_nc: bool = True
    custom_fg_coords_path: Optional[str] = None # For sims w/ multiple kap types, to not redo loading of FGs
    custom_kap_coords_path: Optional[str] = None # For consistency
    categorize_sim_times: Optional[List[str]] = None # Optional list of sim times to merge (e.g. ["0-5", "5-10"]). If None, uses the default time range.
    ignored_simulations: List[int] = field(default_factory=list) # List of simulation indices to ignore starting from stage 3
    
    # --- Stage 4: Embedding ---
    window_size_steps: int = 10 # Window time is WINDOW_SIZE_STEPS * LOAD_MD_STEP_NS
    custom_categorized_path: Optional[str] = None # To skip categorization stage if already done
    
    # --- Stage 5: Clustering ---
    clustering_type: str = "sym-faiss"
    n_clusters: List[int] = field(default_factory=lambda: [40, 80, 160, 320, 640, 1280])
    prune_thresh: float = 0.95 # Threshold for merging heavy nucleoplasm/cytoplasm clusters in sym-kmeans clustering (between 0 and 1, lower means more aggressive merging)
    save_cluster_3d_locations: bool = True # Whether to save 3D locations of clusters (for state choice by distance)
    clustering_z_strength: float = 0.0 # Strength for the Z axis feature in clustering (0 means disabled)
    
    # --- Stage 6: MSM ---
    tm_prior: float = 0
    reversible_msm: bool = False # Use MaximumLikelihoodMSM if True, else BayesianMSM
    
    # --- Stage 7: Stats ---
    use_pcca: bool = False # whether to perform PCCA to coarse-grain the transition matrix
    pcca_macrostates: int = 5 # number of macrostates to coarse-grain to if use_pcca is True
    state_choice_method: str = "prominent" # options: "prominent", "distance", "distance_clustering_z"
    distance_state_threshold_nm: float = 10 # only used if STATE_CHOICE_METHOD is "distance" or "distance_clustering_z"
    # If set, uses a linear-ramp soft-assignment around ±distance_state_threshold_nm instead
    # of a hard threshold.  soft_scale_nm is the ramp width in nm: weight rises linearly
    # from 0 at (threshold − soft_scale_nm) to 1 at threshold (and vice-versa for bottom).
    # A value of ~5–20 nm is a reasonable starting point.
    # Set to None (default) to keep the original hard-threshold behaviour.
    soft_scale_nm: Optional[float] = None
    
    def to_legacy_dict(self):
        """Converts snake_case config to the UPPER_CASE dict expected by stages."""
        return {k.upper(): v for k, v in asdict(self).items()}
    
    def override_from_dict(self, override_dict: dict):
        """Override config parameters from a dictionary."""
        if override_dict is None:
            return
        for key, value in override_dict.items():
            if hasattr(self, key.lower()):
                setattr(self, key.lower(), value)
            else:
                raise KeyError(f"Invalid parameter key: {key}")
    
    
def run(checkpoints_path, start_stage=1, end_stage=7, params_override=None):
    """
    Run the full MSM analysis pipeline:
    1. Load NTR RMF data.
    2. Load FG RMF data.
    3. Categorize trajectories to microstates.
    4. Embed microstates to histograms.
    5. Cluster histograms to mesostates.
    6. Generate mesostate transition matrix.
    7. Calculate stats (Permeability).
    
    Parameters:
    - checkpoints_path (str): Path to save checkpoints and results.
    - start_stage (int): Stage to start from (1-7).
    - end_stage (int): Stage to end at (1-7) (not inclusive).
    - params_override (dict): Dictionary to override default parameters.
    """
    config = iMSMConfig(checkpoints_path=checkpoints_path)
    
    # Override parameters if provided
    config.override_from_dict(params_override)
    
    params = config.to_legacy_dict()
    validate_params(params)

    # Run stages
    stages = {
        1 : stage_01_loadNTRs,
        2 : stage_02_loadFGs,
        3 : stage_03_categorize,
        4 : stage_04_embed,
        5 : stage_05_cluster,
        6 : stage_06_buildMSM,
        7 : stage_07_computePermeabilities
    }
    
    Path(checkpoints_path).mkdir(parents=True, exist_ok=True)
    for i in range(start_stage, end_stage):
        print(f"Running stage {i}...")
        stages[i](params)
        print(f"Stage {i} completed.")

def validate_params(params):
    if params['CLUSTERING_TYPE'] != "sym-faiss":
        raise ValueError("Only 'sym-faiss' clustering type is supported in this pipeline.")
    if params['STATE_CHOICE_METHOD'] not in ["prominent", "distance", "distance_clustering_z"]:
        raise ValueError("STATE_CHOICE_METHOD must be 'prominent', 'distance', or 'distance_clustering_z'.")
    if params['STATE_CHOICE_METHOD'] in ["distance", "distance_clustering_z"] and params['DISTANCE_STATE_THRESHOLD_NM'] <= 0:
        raise ValueError("DISTANCE_STATE_THRESHOLD_NM must be positive if STATE_CHOICE_METHOD is distance-based.")
    if params['DATA_SUBSET_INDEX'] is not None and params['DATA_SUBSET_INDEX'] >= (1 / params['DATA_SUBSET']):
        raise ValueError("DATA_SUBSET_INDEX must be less than 1 / DATA_SUBSET (or None).")
    if params['MODE'] == "bootstrap" and params['BOOTSTRAP_REPEATS'] <= 0:
        raise ValueError("BOOTSTRAP_REPEATS must be greater than 0 when MODE is 'bootstrap'.")
    if params['DATA_SUBSET_MODE'] not in ["time", "simulation"]:
        raise ValueError("DATA_SUBSET_MODE must be either 'time' or 'simulation'.")
    if params['DATA_SUBSET_MODE'] == "simulation" and int(len(params['LOAD_MD_SIMS_RANGE']) * params['DATA_SUBSET']) < 1:
        raise ValueError("DATA_SUBSET is too small for the number of simulations in LOAD_MD_SIMS_RANGE when DATA_SUBSET_MODE is 'simulation'.")

def stage_01_loadNTRs(params):
    from iMSM.extensions.npc.npc_rmf_loaders import multi_load_kap_data
    
    _ensure_dir(_cp(params, "1_single_sim_kap_coords"))
    kap_results = multi_load_kap_data(
        input_rmf_path=params['LOAD_MD_BASE_PATH'],
        kap_radius=params['LOAD_MD_KAP_RADIUS'],
        kap_amount=params['LOAD_MD_KAP_AMOUNT'],
        start_t=params['LOAD_MD_START_TIME_NS'],
        end_t=params['LOAD_MD_END_TIME_NS'],
        step_t=params['LOAD_MD_STEP_NS'],
        sims_range=params['LOAD_MD_SIMS_RANGE'],
        frames_per_file=1,
        one_frame_from_each=True,
        get_nsites=0
    )
    for i, result in enumerate(kap_results):
        if np.count_nonzero(result) == 0:
            sim_index: int = params['LOAD_MD_SIMS_RANGE'][i] if i < len(params['LOAD_MD_SIMS_RANGE']) else i + 1
            raise ValueError(
                f"Loaded KAP coordinates for simulation {sim_index} are completely filled with zeros. "
                f"Checkpoints path: {params['CHECKPOINTS_PATH']}, "
                f"Base RMF path: {params['LOAD_MD_BASE_PATH']}, "
                f"KAP radius: {params['LOAD_MD_KAP_RADIUS']}, "
                f"KAP amount: {params['LOAD_MD_KAP_AMOUNT']}"
            )
        t0_str, t1_str, sim_time_str = _time_range_strings(params)
        sim_dir = _ensure_dir(_cp(params, "1_single_sim_kap_coords", i + 1))
        with open(sim_dir / f"{sim_time_str}.pickle", "wb") as f:
            pickle.dump(result, f)

def stage_02_loadFGs(params):
    from iMSM.extensions.npc.npc_rmf_loaders import multi_load_fg_data, FG_TYPES, N_CHAINS_PER_FG, N_BEADS_PER_FG
    _ensure_dir(_cp(params, "2_single_sim_fg_coords"))
    
    fg_types = FG_TYPES
    n_chains_per_fg = N_CHAINS_PER_FG
    n_beads_per_fg = N_BEADS_PER_FG
    
    for nup_type in params['LOAD_MD_IGNORED_NUP_TYPES']:
        if nup_type in fg_types:
            idx = fg_types.index(nup_type)
            del fg_types[idx]
            del n_chains_per_fg[idx]
            del n_beads_per_fg[idx]
    
    fg_results = multi_load_fg_data(
        input_rmf_path=params['LOAD_MD_BASE_PATH'],
        fg_types=fg_types,
        n_chains_per_fg=n_chains_per_fg,
        n_beads_per_fg=n_beads_per_fg,
        start_t=params['LOAD_MD_START_TIME_NS'],
        end_t=params['LOAD_MD_END_TIME_NS'],
        step_t=params['LOAD_MD_STEP_NS'],
        sims_range=params['LOAD_MD_SIMS_RANGE'],
        frames_per_file=1,
        one_frame_from_each=True,
    )
    for i, result in enumerate(fg_results):
        if np.count_nonzero(result) == 0:
            sim_index: int = params['LOAD_MD_SIMS_RANGE'][i] if i < len(params['LOAD_MD_SIMS_RANGE']) else i + 1
            raise ValueError(
                f"Loaded FG coordinates for simulation {sim_index} are completely filled with zeros. "
                f"Checkpoints path: {params['CHECKPOINTS_PATH']}, "
                f"Base RMF path: {params['LOAD_MD_BASE_PATH']}"
            )
        t0_str, t1_str, sim_time_str = _time_range_strings(params)
        sim_dir = _ensure_dir(_cp(params, "2_single_sim_fg_coords", i + 1))
        with open(sim_dir / f"{sim_time_str}-fgs.pickle", "wb") as f:
            pickle.dump(result, f)

def stage_03_categorize(params):
    from iMSM.extensions.npc.npc_categorize import categorize_multiples
    
    _, _, default_sim_time_str = _time_range_strings(params)
    sim_times = params.get('CATEGORIZE_SIM_TIMES')
    if not sim_times:
        sim_times = [default_sim_time_str]

    fgs_path = (
        params['CUSTOM_FG_COORDS_PATH']
        if params['CUSTOM_FG_COORDS_PATH']
        else str(_cp(params, "2_single_sim_fg_coords")) + "/"
    )
    kaps_path = (
        params['CUSTOM_KAP_COORDS_PATH']
        if params['CUSTOM_KAP_COORDS_PATH']
        else str(_cp(params, "1_single_sim_kap_coords")) + "/"
    )
    categorize_multiples(
        sim_indexes=_get_sim_indexes(params),
        sim_times=sim_times,
        diffuser_coords_path_prefix=kaps_path,
        fg_coords_path_prefix=fgs_path,
        step=1,
        save_file_path=str(_cp(params, "3_categorized.pickle")),
        k=params['INTERACTION_CAPACITY'],
        diffuser_radius_nm=params['LOAD_MD_KAP_RADIUS']/10,
        max_surface_distance_nm=params['MAX_SURFACE_DIST_NM'],
        split_nc=params['SPLIT_NC']
    )

def stage_04_embed(params):
    from iMSM.extensions.npc.npc_embed_cluster import load_embed_save
    
    categorized_path = (
        params['CUSTOM_CATEGORIZED_PATH']
        if params['CUSTOM_CATEGORIZED_PATH']
        else str(_cp(params, "3_categorized.pickle"))
    )
    
    load_embed_save(
    window_size=params['WINDOW_SIZE_STEPS'],
    load_categorized_path=categorized_path,
    save_embedded_path=str(_cp(params, "4_embedded.pickle")),
    save_embedded_eighth_path=None,
    split_nc="nmc"
    )

def stage_05_cluster(params):
    if params['MODE'] == "normal":
        _stage_5_cluster_normal(params)
    elif params['MODE'] == "subset":
        _stage_5_cluster_subset(params)
    else:
        raise ValueError("Invalid MODE for stage 5 (cluster). Must be 'normal' or 'subset'.")

def _stage_5_cluster_subset(params):
    from iMSM.extensions.npc.npc_embed_cluster import load_reduce_cluster_save
    subset = _subset_folder(params)
    _ensure_dir(_cp(params, "5_clustering_subsets", subset))
    _ensure_dir(_cp(params, "5_clustered_subsets", subset))
    _, _, default_sim_time_str = _time_range_strings(params)
    sim_times = params.get('CATEGORIZE_SIM_TIMES')
    if not sim_times:
        sim_times = [default_sim_time_str]

    kaps_path = (
        params['CUSTOM_KAP_COORDS_PATH']
        if params['CUSTOM_KAP_COORDS_PATH']
        else str(_cp(params, "1_single_sim_kap_coords")) + "/"
    )

    load_reduce_cluster_save(
        pca_components=220,
        n_clusters=params['N_CLUSTERS'],
        load_embedded_path=str(_cp(params, "4_embedded.pickle")),
        save_pca_cluster_path=str(_cp(params, "5_clustering_subsets", subset, "#c#clusters.pickle")),
        save_clustered_path=str(_cp(params, "5_clustered_subsets", subset, "#c#clusters.pickle")),
        verbose=False,
        data_subset=params['DATA_SUBSET'],
        data_subset_index=params['DATA_SUBSET_INDEX'],
        data_subset_mode=params['DATA_SUBSET_MODE'],
        n_sims=len(_get_sim_indexes(params)),
        kap_coords_dir=kaps_path,
        sim_indexes=_get_sim_indexes(params),
        sim_time_str=sim_times,
        window_size=params['WINDOW_SIZE_STEPS'],
        prune_thresh=params['PRUNE_THRESH'],
        save_cluster_3d_locations=params['SAVE_CLUSTER_3D_LOCATIONS'],
        clustering_z_strength=params['CLUSTERING_Z_STRENGTH']
        )

def _stage_5_cluster_normal(params):
    from iMSM.extensions.npc.npc_embed_cluster import load_reduce_cluster_save
    _ensure_dir(_cp(params, "5_clustering"))
    _ensure_dir(_cp(params, "5_clustered"))
    _, _, default_sim_time_str = _time_range_strings(params)
    sim_times = params.get('CATEGORIZE_SIM_TIMES')
    if not sim_times:
        sim_times = [default_sim_time_str]

    kaps_path = (
        params['CUSTOM_KAP_COORDS_PATH']
        if params['CUSTOM_KAP_COORDS_PATH']
        else str(_cp(params, "1_single_sim_kap_coords")) + "/"
    )

    load_reduce_cluster_save(
        pca_components=220,
        n_clusters=params['N_CLUSTERS'],
        load_embedded_path=str(_cp(params, "4_embedded.pickle")),
        save_pca_cluster_path=str(_cp(params, "5_clustering", "#c#clusters.pickle")),
        save_clustered_path=str(_cp(params, "5_clustered", "#c#clusters.pickle")),
        verbose=False,
        kap_coords_dir=kaps_path,
        sim_indexes=_get_sim_indexes(params),
        sim_time_str=sim_times,
        window_size=params['WINDOW_SIZE_STEPS'],
        prune_thresh=params['PRUNE_THRESH'],
        save_cluster_3d_locations=params['SAVE_CLUSTER_3D_LOCATIONS'],
        clustering_z_strength=params['CLUSTERING_Z_STRENGTH']
        )

def stage_06_buildMSM(params):
    if params['MODE'] == "subset":
        _stage_06_buildMSM_subset(params)
    else:  
        for n_clusters in params['N_CLUSTERS']:
            with open(_cp(params, "5_clustering", f"{n_clusters}clusters.pickle"), "rb") as f:
                clustering = pickle.load(f)
                actual_n_clusters = clustering.centroids.shape[0]
            with open(_cp(params, "5_clustered", f"{n_clusters}clusters.pickle"), "rb") as f:
                clustered_data = pickle.load(f)
            if params['MODE'] == "normal":
                _stage_06_buildMSM_normal(params, n_clusters, actual_n_clusters, clustered_data)
            elif params['MODE'] == "bootstrap":
                _stage_06_buildMSM_bootstrap(params, n_clusters, actual_n_clusters, clustered_data)
            else:
                raise ValueError("Invalid MODE for stage 6 (MSM). Must be 'normal', 'bootstrap', or 'subset'.")

def _stage_06_buildMSM_subset(params):
    from iMSM.extensions.npc.npc_msm import generate_transition_matrix
    subset = _subset_folder(params)
    out_dir = _ensure_dir(_cp(params, "6_transition_matrices_subsets", subset))
    for n_clusters in params['N_CLUSTERS']:
        with open(_cp(params, "5_clustering_subsets", subset, f"{n_clusters}clusters.pickle"), "rb") as f:
            clustering = pickle.load(f)
            actual_n_clusters = clustering.centroids.shape[0]
        with open(_cp(params, "5_clustered_subsets", subset, f"{n_clusters}clusters.pickle"), "rb") as f:
            clustered_data = pickle.load(f)
        transition_matrix = generate_transition_matrix(clustered_data[:, :], actual_n_clusters, prior=params['TM_PRIOR'], reversible=params.get('REVERSIBLE_MSM', False))
        with open(out_dir / f"{n_clusters}clusters.pickle", "wb") as f:
            pickle.dump(transition_matrix, f)

def _stage_06_buildMSM_normal(params, n_clusters, actual_n_clusters, clustered_data):
    from iMSM.extensions.npc.npc_msm import generate_transition_matrix
    out_dir = _ensure_dir(_cp(params, "6_transition_matrices"))
    transition_matrix = generate_transition_matrix(clustered_data[:, :], actual_n_clusters, prior=params['TM_PRIOR'], reversible=params.get('REVERSIBLE_MSM', False))
    with open(out_dir / f"{n_clusters}clusters.pickle", "wb") as f:
        pickle.dump(transition_matrix, f)

def _stage_06_buildMSM_bootstrap(params, n_clusters, actual_n_clusters, clustered_data):
    from iMSM.extensions.npc.npc_msm import generate_transition_matrix
    out_dir = _ensure_dir(_cp(params, "6_transition_matrices_bootstrap"))
    n_sims = len(_get_sim_indexes(params))
    n_trajs_per_sim = clustered_data.shape[0] // (n_sims * 8) # 8 for eightwise symmetry
    for b in range(params['BOOTSTRAP_REPEATS']):
        # resample sims with replacement
        resampled_indices = np.random.choice(n_sims, n_sims, replace=True)
        resampled_clustered = []
        for sym_i in range(8):
            bottom_range = sym_i * n_sims * n_trajs_per_sim
            sample = np.concatenate([clustered_data[bottom_range + i*n_trajs_per_sim:bottom_range + (i+1)*n_trajs_per_sim, :] for i in resampled_indices], axis=0)
            resampled_clustered.append(sample)
        resampled_clustered = np.concatenate(resampled_clustered, axis=0)
        transition_matrix = generate_transition_matrix(resampled_clustered[:, :], actual_n_clusters, prior=params['TM_PRIOR'], reversible=params.get('REVERSIBLE_MSM', False))
        with open(out_dir / f"{n_clusters}clusters_bootstrap{b+1}.pickle", "wb") as f:
            pickle.dump(transition_matrix, f)
    
    
# def mm_permiability(tm, bottom_states=None, top_states=None):
#     from iMSM.extensions.npc.npc_utils import amount_to_concentration, markov_rate_flux, markov_rate_manual

#     lagtime = 1e-6 # in seconds
#     try:
#         rate_1 = markov_rate_flux(tm, start_states=bottom_states, target_states=top_states) # units: 1/lagtime (probably 1000ns)
#         rate_2 = markov_rate_flux(tm, start_states=top_states, target_states=bottom_states) # units: 1/lagtime (probably 1000ns)
#     except ValueError as e:
#         print(f"Error calculating rates: {e}")
#         return 0
#     rate_s = (1/lagtime) * (rate_1 + rate_2) # units: 1/s

#     concentration_M = amount_to_concentration(1, box_side_a=800) # single molecule
#     concentration_uM = concentration_M * 1e6
#     # print(concentration_uM)
    
#     # units : n_events / s / uM / NPC 
#     perm = rate_s / concentration_uM
    
#     return perm

def mm_permeability_trajectory(tm, bottom_states, top_states,
                               n_trajectories=3000, traj_length=1000,
                               lagtime_s=1e-6):
    """
    Permeability estimate via Markov chain trajectory simulation.

    Counts transport events (bottom→top or top→bottom crossings) along
    simulated trajectories.  Supports both hard and soft state assignments:

    Hard assignment (original behaviour)
    -------------------------------------
    Pass integer index arrays for bottom_states / top_states.  A transport
    event adds 1 to the counter each time the trajectory commits to the
    opposite endpoint set.

    Soft assignment (sigmoid weights)
    ----------------------------------
    Pass float arrays of shape (n_states,) with values in [0, 1] — e.g. the
    output of get_top_bottom_states(..., soft_scale_nm=...).  Each crossing
    contributes ``last_weight × current_weight`` instead of 1, smoothly
    down-weighting states near the threshold boundary.

    Parameters
    ----------
    tm : np.ndarray, shape (n, n)
        Row-stochastic transition matrix.
    bottom_states : array-like of int **or** np.ndarray of float
        Hard indices *or* per-state soft weights for the bottom endpoint set.
    top_states : array-like of int **or** np.ndarray of float
        Hard indices *or* per-state soft weights for the top endpoint set.
    n_trajectories : int
        Number of independent Markov chain trajectories to simulate.
    traj_length : int
        Number of steps per trajectory.
    lagtime_s : float
        Lagtime per step in seconds (default 1e-6 = 1 µs).

    Returns
    -------
    float
        Permeability in units of events / s / µM / NPC.
    """
    from iMSM.extensions.npc.npc_utils import amount_to_concentration

    n_states = tm.shape[0]

    # ------------------------------------------------------------------ #
    # Resolve bottom / top descriptors into per-state weight arrays.       #
    # Float arrays → soft weights (passed through directly).              #
    # Integer arrays → hard 0/1 weights.                                  #
    # ------------------------------------------------------------------ #
    _bot = np.asarray(bottom_states)
    _top = np.asarray(top_states)
    if _bot.dtype.kind == 'f':
        # Soft weights provided directly (shape must be (n_states,))
        bottom_w = _bot
        top_w    = _top
    else:
        # Hard integer indices → binary weight arrays
        bottom_w = np.zeros(n_states)
        bottom_w[_bot.astype(int)] = 1.0
        top_w = np.zeros(n_states)
        top_w[_top.astype(int)] = 1.0

    # Stationary distribution for trajectory initialisation
    eigenvalues, eigenvectors = np.linalg.eig(tm.T)
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    stat_dist = np.real(eigenvectors[:, idx])
    stat_dist = np.abs(stat_dist)
    stat_dist /= stat_dist.sum()

    # Cumulative-sum rows for O(log n) state sampling; clip last column to 1
    cum_tm = np.cumsum(tm, axis=1)
    cum_tm[:, -1] = 1.0

    total_transports = 0.0  # float to accumulate weighted crossings

    for _ in range(n_trajectories):
        state = int(np.random.choice(n_states, p=stat_dist))
        last_side   = None   # None, 0 (bottom), or 1 (top)
        last_weight = 0.0    # soft weight of the last committed state

        for _ in range(traj_length):
            w_b = bottom_w[state]
            w_t = top_w[state]

            # Assign to whichever side has higher weight, but only if that
            # weight exceeds 0.5 — the natural boundary cutoff.
            # For soft assignment: sigmoid > 0.5 iff the state is beyond the
            # threshold distance, independent of soft_scale_nm.
            # For hard assignment: weights are 0 or 1, so > 0.5 ≡ == 1.0.
            if w_b > w_t and w_b > 0.5:
                current_side   = 0
                current_weight = w_b
            elif w_t > w_b and w_t > 0.5:
                current_side   = 1
                current_weight = w_t
            else:
                current_side = None   # "in-channel"

            if current_side is not None:
                if last_side is None:
                    last_side   = current_side    # initialise
                    last_weight = current_weight
                elif last_side != current_side:
                    # Weighted crossing: product of departure and arrival weights
                    total_transports += last_weight * current_weight
                    last_side   = current_side
                    last_weight = current_weight
                else:
                    # Same side re-entry: update departure weight to the most
                    # recent committed state (physically, this is the last
                    # position before the actual crossing attempt).
                    last_weight = current_weight

            # Advance one Markov step
            state = int(np.searchsorted(cum_tm[state], np.random.random()))

    total_time_s     = n_trajectories * traj_length * lagtime_s
    concentration_M  = amount_to_concentration(1, box_side_a=800)
    concentration_uM = concentration_M * 1e6

    rate = total_transports / total_time_s
    return rate / concentration_uM


def get_top_bottom_states(clustering, state_choice_method, distance_state_threshold_nm,
                          clustered=None, soft_scale_nm=None, clusters_3d_locations=None):
    """Return (bottom, top) state descriptors.

    For hard-threshold methods ("prominent", "nuc_cyt_treshold", or "distance" with
    soft_scale_nm=None) the return values are integer index arrays.

    For the "distance" method with soft_scale_nm set, the return values are float
    numpy arrays of shape (n_states,) with values in [0, 1] representing linear-ramp
    soft-assignment weights for each state:

        w_top[i]    = clip( (mu_z[i]  − (threshold − soft_scale_nm)) / soft_scale_nm, 0, 1 )
        w_bottom[i] = clip( (−mu_z[i] − (threshold − soft_scale_nm)) / soft_scale_nm, 0, 1 )

    Weight is 0 at (threshold − soft_scale_nm) and reaches 1 at threshold; states beyond
    the threshold are clamped to 1.  soft_scale_nm is the ramp width in nm.
    """
    bottom_indices = []
    top_indices = []
    if state_choice_method == "prominent":
        if clustered is None:
            raise ValueError("clustered data must be provided for 'prominent' state choice method.")
        unique, counts = np.unique(clustered, return_counts=True)
        top_2_indices = np.argsort(counts)[-2:][::-1]
        most_prominent = unique[top_2_indices]
        if len(most_prominent) < 2:
            raise ValueError("Not enough unique states to determine top and bottom states.")
        bottom_indices.append(most_prominent[0])
        top_indices.append(most_prominent[1])
    elif state_choice_method == "distance":
        from iMSM.extensions.npc.npc_graph_figure import estimate_cluters_mu_cov, calc_coordinate_edges_dict, estimate_clusters_mu_2
        from iMSM.extensions.npc.npc_utils import get_sorted_anchor_coordinates_np
        anchor_coordinates = get_sorted_anchor_coordinates_np()[1]
        coordinate_edges = calc_coordinate_edges_dict(anchor_coordinates)
        mus, covs = estimate_cluters_mu_cov(2000, clustering.centroids, coordinate_edges, range(clustering.centroids.shape[0]))
        # mus = estimate_clusters_mu_2(clusters_3d_locations, range(clustering.centroids.shape[0]))
        if soft_scale_nm is not None:
            # Linear-ramp weights: 0 at (threshold − soft_scale_nm), 1 at threshold.
            top_indices    = np.clip((mus[:, 1]  - (distance_state_threshold_nm - soft_scale_nm)) / soft_scale_nm, 0.0, 1.0)
            bottom_indices = np.clip((-mus[:, 1] - (distance_state_threshold_nm - soft_scale_nm)) / soft_scale_nm, 0.0, 1.0)
            # print(f"Linear-ramp weights (bottom): {bottom_indices}")
            # print(f"Linear-ramp weights (top): {top_indices}")
        else:
            bottom_indices = np.where(mus[:, 1] <= -distance_state_threshold_nm)[0]
            top_indices = np.where(mus[:, 1] >= distance_state_threshold_nm)[0]
    elif state_choice_method == "nuc_cyt_treshold":
        bottom_indices = np.where(clustering.centroids[:,0] > distance_state_threshold_nm)[0]
        top_indices = np.where(clustering.centroids[:,-1] > distance_state_threshold_nm)[0]
    elif state_choice_method == "distance_clustering_z":
        if not hasattr(clustering, 'centroids_z_actual'):
            raise ValueError("clustering does not have centroids_z_actual. Ensure clustering_z_strength > 0 was used.")
        mus_z = clustering.centroids_z_actual.flatten()
        if soft_scale_nm is not None:
            top_indices    = np.clip((mus_z  - (distance_state_threshold_nm - soft_scale_nm)) / soft_scale_nm, 0.0, 1.0)
            bottom_indices = np.clip((-mus_z - (distance_state_threshold_nm - soft_scale_nm)) / soft_scale_nm, 0.0, 1.0)
        else:
            bottom_indices = np.where(mus_z <= -distance_state_threshold_nm)[0]
            top_indices = np.where(mus_z >= distance_state_threshold_nm)[0]
    return bottom_indices, top_indices
            
def stage_07_computePermeabilities(params):
    if params['MODE'] == "normal":
        _stage_07_computePermeabilities_normal(params)
    elif params['MODE'] == "bootstrap":
        _stage_07_computePermeabilities_bootstrap(params)
    elif params['MODE'] == "subset":
        _stage_07_computePermeabilities_subset(params)
    else:    
        raise ValueError("Invalid MODE for stage 7. Must be 'normal' or 'bootstrap'.")



def _stage_07_computePermeabilities_bootstrap(params):
    permeabilities_bootstrap = [{} for _ in range(params['BOOTSTRAP_REPEATS'])]
    for n_clusters in params['N_CLUSTERS']:
        with open(_cp(params, "5_clustered", f"{n_clusters}clusters.pickle"), "rb") as f:
            clustered = pickle.load(f)
        with open(_cp(params, "5_clustering", f"{n_clusters}clusters.pickle"), "rb") as f:
            clustering = pickle.load(f)
        with open(_cp(params, "5_clustering", f"{n_clusters}clusters_3d_locations.pickle"), "rb") as f:
            clusters_3d_locations = pickle.load(f)
        if not params.get('USE_PCCA', False):
            bottom_indices, top_indices = get_top_bottom_states(clustering=clustering,
                                                        state_choice_method=params['STATE_CHOICE_METHOD'],
                                                        distance_state_threshold_nm=params['DISTANCE_STATE_THRESHOLD_NM'],
                                                        clustered=clustered,
                                                        soft_scale_nm=params.get('SOFT_SCALE_NM'),
                                                        clusters_3d_locations=clusters_3d_locations)    
        permeabilities = {}
        for b in range(params['BOOTSTRAP_REPEATS']):
            with open(_cp(params, "6_transition_matrices_bootstrap", f"{n_clusters}clusters_bootstrap{b+1}.pickle"), "rb") as f:
                tm = pickle.load(f)
            
            if params.get('USE_PCCA', False):
                import deeptime
                model = deeptime.markov.msm.MarkovStateModel(tm)
                pcca = model.pcca(params.get('PCCA_MACROSTATES', 5))
                tm = pcca.coarse_grained_transition_matrix
                memberships = pcca.memberships # shape: (n_microstates, n_macrostates)
                
                if params.get('STATE_CHOICE_METHOD') == 'distance_clustering_z':
                    micro_mus_z = clustering.centroids_z_actual.flatten()
                    macro_mus_z = (memberships.T @ micro_mus_z) / memberships.sum(axis=0)
                else:
                    from iMSM.extensions.npc.npc_graph_figure import estimate_cluters_mu_cov, calc_coordinate_edges_dict
                    from iMSM.extensions.npc.npc_utils import get_sorted_anchor_coordinates_np
                    anchor_coordinates = get_sorted_anchor_coordinates_np()[1]
                    coordinate_edges = calc_coordinate_edges_dict(anchor_coordinates)
                    mus, _ = estimate_cluters_mu_cov(2000, clustering.centroids, coordinate_edges, range(clustering.centroids.shape[0]))
                    macro_mus = (memberships.T @ mus) / memberships.sum(axis=0)[:, None]
                    macro_mus_z = macro_mus[:, 1]
                
                if params.get('SOFT_SCALE_NM') is not None:
                    soft_scale_nm = params['SOFT_SCALE_NM']
                    distance_state_threshold_nm = params['DISTANCE_STATE_THRESHOLD_NM']
                    top_indices    = np.clip((macro_mus_z  - (distance_state_threshold_nm - soft_scale_nm)) / soft_scale_nm, 0.0, 1.0)
                    bottom_indices = np.clip((-macro_mus_z - (distance_state_threshold_nm - soft_scale_nm)) / soft_scale_nm, 0.0, 1.0)
                else:
                    bottom_indices = np.where(macro_mus_z <= -params['DISTANCE_STATE_THRESHOLD_NM'])[0]
                    top_indices = np.where(macro_mus_z >= params['DISTANCE_STATE_THRESHOLD_NM'])[0]

            # perm = mm_permiability(tm, bottom_states=bottom_indices, top_states=top_indices)
            perm = mm_permeability_trajectory(tm, bottom_states=bottom_indices, top_states=top_indices, lagtime_s=(1e-9 * params['LOAD_MD_STEP_NS'] * params['WINDOW_SIZE_STEPS']))
            permeabilities_bootstrap[b][n_clusters] = perm
            if b == 0:
                print(f"Bootstrap {b+1} Permeability for {n_clusters} clusters: {perm} (units: n_events / s / uM / NPC)")
    with open(_cp(params, "7_permeabilities_bootstrap.pickle"), "wb") as f:
        pickle.dump(permeabilities_bootstrap, f)

def _stage_07_computePermeabilities_normal(params):
    permeabilities = {}
    for n_clusters in params['N_CLUSTERS']:
        with open(_cp(params, "5_clustered", f"{n_clusters}clusters.pickle"), "rb") as f:
            clustered = pickle.load(f)
        with open(_cp(params, "5_clustering", f"{n_clusters}clusters.pickle"), "rb") as f:
            clustering = pickle.load(f)
        with open(_cp(params, "6_transition_matrices", f"{n_clusters}clusters.pickle"), "rb") as f:
            tm = pickle.load(f)
        with open(_cp(params, "5_clustering", f"{n_clusters}clusters_3d_locations.pickle"), "rb") as f:
                clusters_3d_locations = pickle.load(f)
        if params.get('USE_PCCA', False):
            import deeptime
            model = deeptime.markov.msm.MarkovStateModel(tm)
            pcca = model.pcca(params.get('PCCA_MACROSTATES', 5))
            tm = pcca.coarse_grained_transition_matrix
            memberships = pcca.memberships # shape: (n_microstates, n_macrostates)
            
            if params.get('STATE_CHOICE_METHOD') == 'distance_clustering_z':
                micro_mus_z = clustering.centroids_z_actual.flatten()
                macro_mus_z = (memberships.T @ micro_mus_z) / memberships.sum(axis=0)
            else:
                from iMSM.extensions.npc.npc_graph_figure import estimate_cluters_mu_cov, calc_coordinate_edges_dict, estimate_clusters_mu_2
                from iMSM.extensions.npc.npc_utils import get_sorted_anchor_coordinates_np
                anchor_coordinates = get_sorted_anchor_coordinates_np()[1]
                coordinate_edges = calc_coordinate_edges_dict(anchor_coordinates)
                mus, _ = estimate_cluters_mu_cov(2000, clustering.centroids, coordinate_edges, range(clustering.centroids.shape[0]))
                
                # macro_mus = weighted average of micro_mus
                macro_mus = (memberships.T @ mus) / memberships.sum(axis=0)[:, None]
                macro_mus_z = macro_mus[:, 1]
            
            if params.get('SOFT_SCALE_NM') is not None:
                soft_scale_nm = params['SOFT_SCALE_NM']
                distance_state_threshold_nm = params['DISTANCE_STATE_THRESHOLD_NM']
                top_indices    = np.clip((macro_mus_z  - (distance_state_threshold_nm - soft_scale_nm)) / soft_scale_nm, 0.0, 1.0)
                bottom_indices = np.clip((-macro_mus_z - (distance_state_threshold_nm - soft_scale_nm)) / soft_scale_nm, 0.0, 1.0)
            else:
                bottom_indices = np.where(macro_mus_z <= -params['DISTANCE_STATE_THRESHOLD_NM'])[0]
                top_indices = np.where(macro_mus_z >= params['DISTANCE_STATE_THRESHOLD_NM'])[0]
        else:
            bottom_indices, top_indices = get_top_bottom_states(clustering=clustering,
                                                                state_choice_method=params['STATE_CHOICE_METHOD'],
                                                                distance_state_threshold_nm=params['DISTANCE_STATE_THRESHOLD_NM'],
                                                                clustered=clustered,
                                                                soft_scale_nm=params.get('SOFT_SCALE_NM'),
                                                                clusters_3d_locations=clusters_3d_locations)
        # perm = mm_permiability(tm, bottom_states=bottom_indices, top_states=top_indices)
        perm = mm_permeability_trajectory(tm, bottom_states=bottom_indices, top_states=top_indices, lagtime_s=(1e-9 * params['LOAD_MD_STEP_NS'] * params['WINDOW_SIZE_STEPS']))
        permeabilities[n_clusters] = perm
        print(f"Permeability for {n_clusters} clusters: {perm} (units: n_events / s / uM / NPC)")
    with open(_cp(params, "7_permeabilities.pickle"), "wb") as f:
        pickle.dump(permeabilities, f)
        
        
def _stage_07_computePermeabilities_subset(params):
    subset = _subset_folder(params)
    out_dir = _ensure_dir(_cp(params, "7_permeabilities_subsets", subset))
    permeabilities = {}
    for n_clusters in params['N_CLUSTERS']:
        with open(_cp(params, "5_clustered_subsets", subset, f"{n_clusters}clusters.pickle"), "rb") as f:
            clustered = pickle.load(f)
        with open(_cp(params, "5_clustering_subsets", subset, f"{n_clusters}clusters.pickle"), "rb") as f:
            clustering = pickle.load(f)
        with open(_cp(params, "6_transition_matrices_subsets", subset, f"{n_clusters}clusters.pickle"), "rb") as f:
            tm = pickle.load(f)
        with open(_cp(params, "5_clustering_subsets", subset, f"{n_clusters}clusters_3d_locations.pickle"), "rb") as f:
            clusters_3d_locations = pickle.load(f)
        if params.get('USE_PCCA', False):
            import deeptime
            model = deeptime.markov.msm.MarkovStateModel(tm)
            pcca = model.pcca(params.get('PCCA_MACROSTATES', 5))
            tm = pcca.coarse_grained_transition_matrix
            memberships = pcca.memberships # shape: (n_microstates, n_macrostates)
            
            if params.get('STATE_CHOICE_METHOD') == 'distance_clustering_z':
                micro_mus_z = clustering.centroids_z_actual.flatten()
                macro_mus_z = (memberships.T @ micro_mus_z) / memberships.sum(axis=0)
            else:
                from iMSM.extensions.npc.npc_graph_figure import estimate_cluters_mu_cov, calc_coordinate_edges_dict, estimate_clusters_mu_2
                from iMSM.extensions.npc.npc_utils import get_sorted_anchor_coordinates_np
                anchor_coordinates = get_sorted_anchor_coordinates_np()[1]
                coordinate_edges = calc_coordinate_edges_dict(anchor_coordinates)
                mus, _ = estimate_cluters_mu_cov(2000, clustering.centroids, coordinate_edges, range(clustering.centroids.shape[0]))
                
                # macro_mus = weighted average of micro_mus
                macro_mus = (memberships.T @ mus) / memberships.sum(axis=0)[:, None]
                macro_mus_z = macro_mus[:, 1]
            
            if params.get('SOFT_SCALE_NM') is not None:
                soft_scale_nm = params['SOFT_SCALE_NM']
                distance_state_threshold_nm = params['DISTANCE_STATE_THRESHOLD_NM']
                top_indices    = np.clip((macro_mus_z  - (distance_state_threshold_nm - soft_scale_nm)) / soft_scale_nm, 0.0, 1.0)
                bottom_indices = np.clip((-macro_mus_z - (distance_state_threshold_nm - soft_scale_nm)) / soft_scale_nm, 0.0, 1.0)
            else:
                bottom_indices = np.where(macro_mus_z <= -params['DISTANCE_STATE_THRESHOLD_NM'])[0]
                top_indices = np.where(macro_mus_z >= params['DISTANCE_STATE_THRESHOLD_NM'])[0]
        else:
            bottom_indices, top_indices = get_top_bottom_states(clustering=clustering,
                                                                    state_choice_method=params['STATE_CHOICE_METHOD'],
                                                                    distance_state_threshold_nm=params['DISTANCE_STATE_THRESHOLD_NM'],
                                                                    clustered=clustered,
                                                                    soft_scale_nm=params.get('SOFT_SCALE_NM'),
                                                                    clusters_3d_locations=clusters_3d_locations)
        perm = mm_permeability_trajectory(tm, bottom_states=bottom_indices, top_states=top_indices, lagtime_s=(1e-9 * params['LOAD_MD_STEP_NS'] * params['WINDOW_SIZE_STEPS']))
        permeabilities[n_clusters] = perm
        print(f"Permeability for {n_clusters} clusters: {perm} (units: n_events / s / uM / NPC)")
    with open(out_dir / "7_permeabilities.pickle", "wb") as f:
        pickle.dump(permeabilities, f)
        
        

