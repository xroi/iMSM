import numpy as np

from iMSM.data_classes.categorization_data_classes import iMSMCategorization
from iMSM.data_classes.embedding_data_classes import iMSMEmbedding, iMSMSingleSimEmbedding
from iMSM.data_classes.config_data_class import iMSMConfig


def divide_to_sections(categorized_trajectory, window_size):
    """
    categorized_trajectories: shape [interaction_capacity, time]
    return: shape [interaction_capacity * window_size, n_sections(=time//window_size)]
    """
    n_closest, n_time = categorized_trajectory.shape
    n_sections = n_time // window_size
    categorized_trajectory = categorized_trajectory[:, :n_sections * window_size]
    divided_trajectories = np.zeros((n_closest * window_size, n_sections), dtype='U20') # string array
    for i in range(n_sections):
        divided_trajectories[:, i] = categorized_trajectory[:, i * window_size:(i + 1) * window_size].flatten()
    return divided_trajectories
    
def multi_divide_to_sections(categorized_trajectories, window_size):
    """
    categorized_trajectories: shape [n_diffusers, interaction_capacity, time]
    return: shape [n_diffusers, interaction_capacity * window_size, n_sections(=time//window_size)]
    """
    n_diffusers, n_closest, n_time = categorized_trajectories.shape
    divided_trajectories = np.zeros((n_diffusers, n_closest * window_size, n_time // window_size), dtype='U20') # string array
    for i in range(n_diffusers):
        divided_trajectories[i] = divide_to_sections(categorized_trajectories[i], window_size)
    return divided_trajectories

def embed_section(section, state_to_idx_dict, n_unique_components):
    """
    return: shape [n_unique_components]
    """
    indexed_section = np.zeros_like(section, dtype=int)
    for i, state in enumerate(section):
        indexed_section[i] = state_to_idx_dict[state]
    return np.bincount(indexed_section, minlength=n_unique_components) / len(indexed_section)

def default_embed(categorization: iMSMCategorization, iMSMConfig: iMSMConfig) -> iMSMEmbedding:
    window_size = iMSMConfig.window_size
    unique_components = np.unique(categorization.components)
    # remove focal component from unique components
    unique_components = unique_components[unique_components != categorization.focal_component]
    n_unique_components = len(unique_components)
    state_to_idx_dict = {state: idx for idx, state in enumerate(unique_components)}
    
    # add unbound state
    state_to_idx_dict['Unbound'] = n_unique_components
    n_unique_components += 1
    
    categorized_single_sim = categorization.trajectories
    embedded_trajectories = []
    
    # todo potentially parallelize this loop (its fast though)
    for single_sim_categorized in categorized_single_sim:
        categorized_trajectory = single_sim_categorized.trajectory # shape [N_focal, interaction_capacity, N_time]
        divided_trajectories = multi_divide_to_sections(categorized_trajectory, window_size)
        embedded_sections = np.zeros((divided_trajectories.shape[0], n_unique_components, divided_trajectories.shape[2])) # (n_diffusers, n_unique_components, n_sections)
        for i_diffuser in range(divided_trajectories.shape[0]):
            for i_section in range(divided_trajectories.shape[2]):
                # embed the section
                section = divided_trajectories[i_diffuser, :, i_section]
                section = embed_section(section, state_to_idx_dict, n_unique_components)
                embedded_sections[i_diffuser, :, i_section] = section
        embedded_trajectories.append(iMSMSingleSimEmbedding(trajectory=embedded_sections))
    
    return iMSMEmbedding(trajectories=embedded_trajectories, unique_components=np.array(list(state_to_idx_dict.keys())))