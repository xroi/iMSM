import numpy as np
from dataclasses import dataclass

@dataclass
class iMSMSingleSimCategorization:
    trajectory: np.ndarray[np.str_] # N_focal_component x interaction_capacity x N_timepoints
    
@dataclass
class iMSMCategorization:
    trajectories: list[iMSMSingleSimCategorization] # N_simulations
    components: np.ndarray[np.str_] # N_components
    focal_component: str
    
    