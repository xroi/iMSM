import numpy as np
import concurrent.futures
from scipy.spatial import cKDTree

from iMSM.data_classes.input_data_classes import iMSMInput, iMSMInputSingleSim
from iMSM.data_classes.categorization_data_classes import iMSMCategorization, iMSMSingleSimCategorization
from iMSM.data_classes.config_data_class import iMSMConfig

def _process_single_simulation(
    sim_data: iMSMInputSingleSim, 
    components: np.ndarray, 
    focal_component: str, 
    max_dist: float,
    interaction_capacity: int
) -> iMSMSingleSimCategorization:
    """
    Worker function to process a single simulation trajectory.
    Returns array of shape: [N_focal, interaction_capacity, N_time]
    """
    # 1. Identify Focal and Environment indices
    is_focal = np.char.startswith(components, focal_component)
    all_indices = np.arange(len(components))
    
    focal_indices = all_indices[is_focal]
    env_indices = all_indices[~is_focal]
    
    env_labels = components[env_indices]
    
    # Shape: [N_comps, 3, N_time]
    focal_traj = sim_data.trajectory[focal_indices, :, :]
    env_traj = sim_data.trajectory[env_indices, :, :]
    
    n_focal = focal_traj.shape[0]
    n_time = focal_traj.shape[2]
    
    # Initialize result array with "Unbound"
    # Shape Change: Now 3D array [N_focal, interaction_capacity, N_time]
    categorized_traj = np.full((n_focal, interaction_capacity, n_time), "Unbound", dtype='U30')
    
    # If no environment exists, return all Unbound immediately
    if len(env_indices) == 0:
        return iMSMSingleSimCategorization(trajectory=categorized_traj)

    # Loop over time
    for t in range(n_time):
        xyz_env = env_traj[:, :, t]   # [N_env, 3]
        xyz_focal = focal_traj[:, :, t] # [N_focal, 3]
        
        # Build KDTree on Environment beads
        tree = cKDTree(xyz_env)
        
        # Query closest neighbors (k=interaction_capacity) within max_dist
        # dists and idxs shape: [N_focal, interaction_capacity] (if interaction_capacity > 1)
        # If interaction_capacity=1, cKDTree returns [N_focal], so we reshape to ensure consistency
        dists, idxs = tree.query(xyz_focal, k=interaction_capacity, distance_upper_bound=max_dist)
        
        # Ensure shape is [N_focal, interaction_capacity] even if k=1
        if interaction_capacity == 1:
            dists = dists[:, np.newaxis]
            idxs = idxs[:, np.newaxis]

        # Filter valid hits (distance < inf)
        # We process the results directly into the categorization array
        valid_mask = (dists != np.inf)
        
        # If there are any valid hits
        if np.any(valid_mask):
            # We can use boolean indexing if we are careful, or simply iterate 
            # over the 'k' dimension which is usually small.
            # Using masking for efficiency:
            
            # idxs contains indices into the env_labels array
            # We only access idxs where valid_mask is True to avoid out-of-bounds 
            # (cKDTree returns n_env for invalid hits)
            
            valid_indices = idxs[valid_mask]
            
            # Map valid neighbor indices to their labels
            valid_labels = env_labels[valid_indices]
            
            # Assign to the correct positions in the 3D array
            # We need to construct a mask for the categorized_traj[:, :, t] slice
            categorized_traj[:, :, t][valid_mask] = valid_labels

    return iMSMSingleSimCategorization(trajectory=categorized_traj)


def default_categorize(iMSMInput: iMSMInput, iMSMConfig: iMSMConfig) -> iMSMCategorization:
    max_surface_dist = iMSMConfig.max_surface_dist
    n_cpus = iMSMConfig.n_cpus
    interaction_capacity = iMSMConfig.interaction_capacity
    
    focal_comp = iMSMInput.focal_component
    comps = iMSMInput.components
    trajectories = iMSMInput.trajectories
    
    results = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_cpus) as executor:
        # Submit tasks
        future_to_sim_idx = {
            executor.submit(
                _process_single_simulation, 
                sim, 
                comps, 
                focal_comp, 
                max_surface_dist,
                interaction_capacity  # Pass interaction_capacity to the worker
            ): i for i, sim in enumerate(trajectories)
        }
        
        # Collect results
        for future in concurrent.futures.as_completed(future_to_sim_idx):
            sim_idx = future_to_sim_idx[future]
            try:
                data = future.result()
                results.append((sim_idx, data))
            except Exception as e:
                print(f"Simulation {sim_idx} failed: {e}")
                raise e

    # Restore original order
    results.sort(key=lambda x: x[0])
    ordered_trajectories = [r[1] for r in results]

    return iMSMCategorization(
        trajectories=ordered_trajectories,
        components=comps,
        focal_component=focal_comp
    )