from typing import Optional, Callable
from attr import asdict, field
from dataclasses import dataclass
from iMSM.data_classes.input_data_classes import iMSMInput
from iMSM.data_classes.categorization_data_classes import iMSMCategorization
from iMSM.data_classes.embedding_data_classes import iMSMEmbedding
from iMSM.data_classes.clustering_data_classes import iMSMClustering
from iMSM.data_classes.msm_data_classes import iMSMMSM
import numpy as np

@dataclass
class iMSMConfig:
    # --- General ---
    checkpoints_path: str
    mode: str = "normal" #options: "normal", "bootstrap", "subset"
    start_stage: int = 1
    end_stage: int = 5
    n_cpus: int = 1
    seed: Optional[int] = None
    
    
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
    
    # --- Stage 1: Categorization ---
    interaction_capacity: int = 5
    max_surface_dist: float = 1.0
    custom_categorization: Optional[Callable[[iMSMInput], iMSMCategorization]] = None
    
    # --- Stage 2: Embedding ---
    window_size: int = 10 # Window time is WINDOW_SIZE * LOAD_MD_STEP_NS
    custom_embedding: Optional[Callable[[iMSMCategorization], iMSMEmbedding]] = None
    
    # --- Stage 3: Clustering ---
    n_clusters: int = 100
    merge_cluster_threshold: Optional[float] = None
    custom_clustering: Optional[Callable[[iMSMEmbedding], iMSMClustering]] = None
    
    # --- Stage 4: MSM Construction ---
    tm_prior: float = 0
    custom_MSM_construction: Optional[Callable[[iMSMClustering], iMSMMSM]] = None
    
    def to_legacy_dict(self):
        return {k.upper(): v for k, v in asdict(self).items()}
    
    def override_from_dict(self, override_dict: dict):
        if override_dict is None:
            return
        for key, value in override_dict.items():
            if hasattr(self, key.lower()):
                setattr(self, key.lower(), value)
            else:
                raise KeyError(f"Invalid parameter key: {key}")

