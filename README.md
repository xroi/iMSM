# iMSM: Interaction-Based Markov State Models

**iMSM** is a computational framework for constructing interpretable Markov state models for 'fuzzy' biomolecular assemblies in which molecules continually form and break contacts. These models describe recurring interaction states and how the system moves between them. The key contribution is a way to represent a molecule's changing interaction environment: summarize which partners it contacts and how extensively it is engaged over short time windows. iMSM learns states from these interaction profiles, turning rapid contact exchange into a compact network whose states and transitions have a direct molecular interpretation.

This repository accompanies the manuscript:
> **Learning interpretable kinetic models for biomolecular interaction networks**  
> *Roi Eliasian, Yael Hazan, Timna Tzori, and Barak Raveh*  
> *(In review)*

---

## General Applicability & Method Overview

iMSMs automate the identification of recurring interaction states and the estimation of their transition networks after the user chooses which molecular components to follow and what counts as an interaction.

### When Can iMSM Be Applied?
The framework is not tied a specific system or a particular level of molecular detail.  It is broadly applicable to molecular dynamics (MD) trajectories using either atomistic or coarse-grained models. It can be applied whenever:
- The dynamics of interest can be represented by **recurring local interaction environments** around a chosen **focal entity**: the part of the system whose interactions you follow. This can be a selected group of atoms or residues, a domain, a molecule, or an entire molecular assembly. Examples include residues involved in transient binding between two proteins or a molecular complex within a larger dynamic biomolecular network, such as a condensate.

- The transitions between these interaction environments are approximately Markovian at an appropriate lag time $\tau$ (the next-state probabilities depend mainly on the current state rather than its history). This approximation is often reasonable in practice and can be assessed using standard kinetic validation tests.

By discretizing time-averaged interaction distributions, iMSMs coarse-grain over microscopic contact configurations that need not share the same local Cartesian geometry or contact network, provided they preserve the same statistical interaction pattern around the focal entity over a certain time window. Different molecular configurations can therefore belong to the same state if they have similar time-averaged patterns of interaction partners and degree of engagement. Examples include:
- **Translocation & Crowding**: Characterizing how the interaction environment of a cargo:transport-receptor complex changes as it traverses the central channel of the nuclear pore complex (NPC).
- **Macromolecular Binding & Recognition**: Distinguishing strongly, partially, and weakly engaged states between interacting binding partners or flexible interfaces.
- **Conformational & Folding Dynamics**: Mapping peptide or disordered protein transitions across intramolecular contact patterns (e.g., Chignolin peptide folding).

In the accompanying manuscript, we applied iMSM at two disparate resolutions: coarse-grained simulations of transport through an entire NPC and atomistic simulations of a protein motif exchanging contacts with a transport receptor.

### 5-Step Workflow
1. **System Partitioning**: The simulated system is partitioned into a set of discrete interacting components, $\mathcal{C} = \{c_1, \ldots, c_N\}$ (e.g., distinct FG Nups, receptor motifs, or residue groups), together with a focal entity $f$ (e.g., a cargo:NTR complex, ligand, or peptide core) that interacts with subsets of these components.
2. **Interaction Trajectory**: Each trajectory is converted into an interaction trajectory $I(t)$ recording contacts between $f$ and $\mathcal{C}$.
3. **Windowed Interaction Histograms**: $I(t)$ is divided into consecutive time windows of duration $\tau$. Interaction statistics in each window are summarized into an interaction histogram $H$ capturing component identities and time-averaged degree of engagement. The histogram is normalized by a fixed reference interaction capacity and includes an entry for unoccupied capacity to distinguish strongly, partially, and weakly engaged states. It can be augmented with process-specific observables.
4. **Unsupervised Clustering**: Unsupervised clustering of the histograms identifies a reduced set of recurring interaction states, $\mathcal{S} = \{s_1, \ldots, s_k\}$, without manually labeling the states.
5. **Lagged Transition Matrix Inference**: Transitions between states are used to infer the lagged transition-probability matrix:
   $$T_{ij} = P\left(s(t+\tau) = s_j \mid s(t) = s_i\right)$$
   from which stationary distributions, implied timescales, transport observables, and pathway fluxes are estimated.

> **Modeling Choices vs. Inferred States**: The focal entity, interacting components, contact criteria, window duration, and clustering resolution are user modeling choices, whereas state assignments and transition probabilities are inferred directly from the trajectories. These choices determine what the model can resolve. Each application requires adequate sampling and validation of the inferred kinetics.

---

## Key Features

- **Modular MSM Pipeline**: End-to-end workflow consisting of four software stages:
  1. `Categorization` — extracts discrete intermolecular or intramolecular interaction signatures across simulation frames.
  2. `Embedding` — converts interaction statistics into numerical representations for clustering.
  3. `Clustering` — identifies recurring interaction states.
  4. `MSM Estimation` — estimates transition rate generators, implied timescales, stationary distributions, and committor probabilities and transition pathways.
- **Nuclear Pore Complex (NPC) Extension**: Specialized tools for analyzing high-throughput nuclear pore complex Brownian dynamics simulations, multi-site FG-repeat sliding, and karyopherin (KAP) translocation.

**Optional symmetry-aware clustering.** iMSMs do not require symmetry. Even dynamic, fuzzy assemblies can have statistically equivalent environments, as in many transport pores. For the NPC, we adapted k-means to exploit this equivalence and improve sampling statistics, calling this symmetry-constrained k-means (`SKM`).[^skm] Future extensions could exploit other symmetries, such as approximate rotational symmetry in spherical condensates where supported by their internal dynamics.

---

## Quickstart & Tutorials

The best way to get started with `iMSM` is through the interactive tutorials in the [`tutorials/`](tutorials/) directory:

| Tutorial | Description |
| :--- | :--- |
| **[`tutorials/1_NPC.ipynb`](tutorials/1_NPC.ipynb)** | **Flagship NPC Transport Pipeline** — End-to-end execution of interaction categorization, embedding, clustering, and kinetic graph construction for nuclear transport trajectories. |
| **[`tutorials/2_chignolin.ipynb`](tutorials/2_chignolin.ipynb)** | **General Biomolecular Dynamics** — Step-by-step walkthrough applying iMSM to the fast-folding Chignolin peptide mini-protein trajectory. |
| **[`tutorials/3_fg_sliding.ipynb`](tutorials/3_fg_sliding.ipynb)** | **FG-Repeat Sliding Dynamics** — Construction of iMSM of FG repeat sliding on Kap95 surface (Figure 6). |

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/xroi/iMSM.git
cd iMSM
```

### 2. Set up the Conda environment
```bash
# Create and activate environment with core scientific libraries
conda create -n imsm python=3.10 numpy scipy pandas matplotlib seaborn mdtraj -c conda-forge -y
conda activate imsm

# Optional dependencies for specialized plotting and generalized additive models:
conda install -c conda-forge pygam -y
```

### 3. Install iMSM in editable mode
```bash
pip install -e .
```

---

## Reproducing Manuscript Figures

Scripts and helper libraries for reproducing the figures and videos in the manuscript are organized in [`figure_scripts/`](figure_scripts/):

- **Figure 2**: [`figure_scripts/figure_2.py`](figure_scripts/figure_2.py) — 4-site vs. 6-site translocation graphs, stationary distributions, and kinetic comparisons.
- **Figure 3**: [`figure_scripts/figure_3.py`](figure_scripts/figure_3.py) — Permeability and free-energy profiles across receptor sizes and valencies, with comparisons between iMSMs and the underlying simulations.
- **Figure 4**: [`figure_scripts/figure_4.py`](figure_scripts/figure_4.py) — Simulation subset convergence analysis and evaluation of gains in sampling efficiency.
- **Figure 5**: [`figure_scripts/figure_5.py`](figure_scripts/figure_5.py) — Free-energy barriers versus transport-receptor molecular weight across dilated pore diameters, with corresponding interaction networks.
- **Figure 6**: [`figure_scripts/figure_6.py`](figure_scripts/figure_6.py) — Atomic-scale FG-repeat sliding dynamics, VMD states, and spatial interaction networks.
- **Supplementary Figures**: [`figure_scripts/figure_supp.py`](figure_scripts/figure_supp.py) — Implied timescale validations, Chapman-Kolmogorov tests, and barrier profiles.
- **Video Animations**: [`figure_scripts/figure_videos.py`](figure_scripts/figure_videos.py) — Generation of side-by-side Brownian dynamics simulation and MSM network videos.

---

## Repository Structure

```
.
├── iMSM/              # Core library (categorize, embed, cluster, msm, NPC extensions)
├── tutorials/         # Getting-started Jupyter notebooks (NPC transport, Chignolin folding, & FG sliding)
├── figure_scripts/    # Reproducible scripts and helpers for manuscript figures
├── data/              # Simulation trajectories, transition matrices, and clustering data
├── plots/             # Rendered output figures, panels, and video artifacts
└── pyproject.toml     # Package configuration
```

---

## Citation

```bibtex
@article{eliasian2026imsm,
  title={Learning interpretable kinetic models for biomolecular interaction networks},
  author={Eliasian, Roi and Hazan, Yael and Tzori, Timna and Raveh, Barak},
  journal={In review},
  year={2026}
}
```

[^skm]: Related, but not identical, clustering approaches: [Charalampidis (2005)](https://doi.org/10.1109/TPAMI.2005.230) and [Mukuta and Harada (2023)](https://arxiv.org/abs/1906.01857).
