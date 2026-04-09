# ATM Network Optimisation — Al-Safwa Bank, Jordan

> **AI-Powered ATM Reduction Under Regulatory and Operational Constraints**  
> Risk Management Department · Data Science & AI Division

---

## Table of Contents

1. [Project Background](#1-project-background)
2. [The Problem (Formally)](#2-the-problem-formally)
3. [AI Architecture Overview](#3-ai-architecture-overview)
4. [Model 1 — Graph Neural Network (GNN)](#4-model-1--graph-neural-network-gnn)
   - 4.1 [Why a Graph?](#41-why-a-graph)
   - 4.2 [Graph Construction](#42-graph-construction)
   - 4.3 [GCN — Graph Convolutional Network](#43-gcn--graph-convolutional-network)
   - 4.4 [GraphSAGE — Graph Sample and Aggregate](#44-graphsage--graph-sample-and-aggregate)
   - 4.5 [Synthetic Label Generation](#45-synthetic-label-generation)
   - 4.6 [Training Details](#46-training-details)
   - 4.7 [What the GNN Learns](#47-what-the-gnn-learns)
5. [Model 2 — Integer Linear Programme (ILP)](#5-model-2--integer-linear-programme-ilp)
   - 5.1 [Why ILP?](#51-why-ilp)
   - 5.2 [Mathematical Formulation](#52-mathematical-formulation)
   - 5.3 [Constraints Explained](#53-constraints-explained)
6. [Model 3 — Scenario Sensitivity Analysis](#6-model-3--scenario-sensitivity-analysis)
7. [Data Pipeline](#7-data-pipeline)
8. [Project Structure](#8-project-structure)
9. [How to Run](#9-how-to-run)
10. [Dashboard Guide](#10-dashboard-guide)
11. [Outputs & Metrics](#11-outputs--metrics)
12. [References](#12-references)

---

## 1. Project Background

The **Central Bank of Jordan** may mandate Al-Safwa Bank to reduce the number of its operational ATMs by **25 % to 50 %** due to:

- Rising **fuel and electricity costs** (powering ATMs 24/7 is expensive)
- Potential **curfew scenarios** caused by regional instability, which force mall branches offline
- Regulatory pressure to rationalise banking infrastructure

Al-Safwa Bank operates **45 ATMs** across **9 governorates**:  
Amman · Zarqa · Irbid · Mafraq · Karak · Balqa (Salt) · Aqaba · Jerash · Madaba

The challenge is not just *how many* to cut — it is *which ones*, so that:
- Geographic coverage of the population is preserved
- Revenue loss is minimised
- Customers are not left without an accessible ATM
- Regulatory constraints (regional coverage, generator-backed power) are satisfied

---

## 2. The Problem (Formally)

This is a **Network Optimisation Under Constraint** problem, specifically a variant of the classic **Budgeted Maximum Coverage Problem**:

```
Given:
  N = 45 ATMs, each with a utility score u_i ∈ [0, 1]
  k = number of ATMs to keep (k = ⌊N × reduction_rate⌋)

Find:
  x ∈ {0,1}^N  such that Σ (u_i × x_i) is maximised
  subject to operational and regulatory constraints
```

This problem is **NP-hard** in general (it is a variant of the 0-1 Knapsack problem). We solve it exactly using Integer Linear Programming after computing the utility scores with a Graph Neural Network.

---

## 3. AI Architecture Overview

The system uses a **three-model pipeline**:

```
ATM_Simulated_Dataset_Safwa.xlsx
            │
            ▼
  ┌─────────────────────┐
  │   Data Pipeline     │  Load · Clean · Normalise · Encode
  └──────────┬──────────┘
             │  Node feature matrix X  (45 × 13)
             │  Distance matrix D      (45 × 45)
             ▼
  ┌─────────────────────┐
  │  Graph Builder      │  Spatial proximity graph G = (V, E)
  │  (Haversine ≤ r km) │  Normalised adjacency matrix A_norm
  └──────────┬──────────┘
             │
             ▼
  ┌─────────────────────┐     Synthetic utility
  │  GNN (GCN / SAGE)   │◄────── labels  y ∈ [0,1]^45
  │  Node regression    │
  └──────────┬──────────┘
             │  Utility scores  û ∈ [0,1]^45
             ▼
  ┌─────────────────────┐
  │  ILP Optimiser      │  Maximise Σ û_i × x_i
  │  (PuLP / CBC)       │  subject to 5 constraint types
  └──────────┬──────────┘
             │  Binary selection  x ∈ {0,1}^45
             ▼
  ┌─────────────────────┐
  │  Scenario Runner    │  4 operational scenarios
  └──────────┬──────────┘
             │
             ▼
  ┌─────────────────────┐
  │  Streamlit Dashboard│  Arabic RTL · Folium map · Export
  └─────────────────────┘
```

---

## 4. Model 1 — Graph Neural Network (GNN)

### 4.1 Why a Graph?

ATMs do not exist in isolation. If ATM A is removed, its customers do not disappear — they migrate to the nearest substitute. A naïve ranking (e.g., sort by transaction count) ignores this network dependency.

**A Graph Neural Network (GNN) models this explicitly:**
- Each ATM is a **node** carrying its own features
- ATMs within a configurable radius (default 5 km) are connected by **edges**
- Edge weights encode proximity: closer ATMs have stronger connections
- During message passing, each node aggregates information from its neighbours

The result: an ATM's utility score reflects both its *own* characteristics **and** its *structural position* in the network — whether it is isolated, whether it sits in a dense cluster, and how unique its services are relative to its neighbours.

---

### 4.2 Graph Construction

**Nodes:** 45 ATMs, indexed 0–44.

**Edge creation:** Two ATMs are connected if their Haversine distance ≤ `radius_km` (default: 5 km).

```
Haversine(lat₁, lon₁, lat₂, lon₂) = 2R · arcsin(√[sin²(Δlat/2) + cos(lat₁)·cos(lat₂)·sin²(Δlon/2)])
```

**Edge weight:** `w(i,j) = 1 / distance_km(i, j)`  
(closer ATMs → stronger graph connection)

**Graph statistics (5 km radius):**
| Metric | Value |
|---|---|
| Nodes | 45 |
| Edges | ~200 |
| Isolated nodes (nearest > 15 km) | 5 (Aqaba, Karak, Jerash, Mafraq, Ramtha) |
| Densest cluster | Central Amman (~25 ATMs within 3 km) |

**Isolated ATMs** (those with zero neighbours within the radius) are automatically protected by the ILP — they cannot be removed because there is no substitute for miles around.

---

### 4.3 GCN — Graph Convolutional Network

The GCN (Kipf & Welling, ICLR 2017) performs **spectral graph convolution** using a normalised adjacency matrix:

**Symmetric normalisation:**
```
Ã = A + I          (add self-loops so each node sees itself)
D̃ = diag(Ã·1)     (degree matrix of Ã)
A_norm = D̃^{-½} · Ã · D̃^{-½}   (symmetric normalisation)
```

**Layer computation:**
```
H^(l+1) = ReLU( A_norm · H^(l) · W^(l) )
```
where `H^(l)` is the node feature matrix at layer `l` and `W^(l)` is a learnable weight matrix.

**Architecture:**
```
Input  →  GCNLayer(13 → 64)  →  Dropout(0.3)
       →  GCNLayer(64 → 32)
       →  Linear(32 → 1)
       →  Sigmoid
Output: utility score ∈ [0, 1] per node
```

**Key property:** Each layer aggregates one hop of neighbours. With 2 layers, each ATM "sees" ATMs up to 2 hops away (up to 10 km). This captures the substitutability pressure — an ATM surrounded by many neighbours inherently becomes less critical.

---

### 4.4 GraphSAGE — Graph Sample and Aggregate

GraphSAGE (Hamilton et al., NeurIPS 2017) is an **inductive** variant that explicitly separates the node's own representation from its neighbourhood summary:

**Layer computation:**
```
h_N(v)^(l) = MEAN({ h_u^(l) : u ∈ N(v) })     (mean aggregation)
h_v^(l+1)  = ReLU( W^(l) · CONCAT( h_v^(l), h_N(v)^(l) ) )
```

**Architecture:**
```
Input  →  SAGEConv(13 → 64)   →  Dropout(0.3)
       →  SAGEConv(64 → 32)
       →  Linear(32 → 1)
       →  Sigmoid
Output: utility score ∈ [0, 1] per node
```

**When to prefer GraphSAGE over GCN:**
- GraphSAGE uses row-normalised adjacency (`D^{-1}A`), better for uneven degree distributions
- It explicitly concatenates self-features with neighbour aggregation, preserving node identity
- For our sparse graph (isolated nodes have degree 0), GraphSAGE degrades gracefully to an MLP for those nodes

Both architectures are available; the default is **GCN**.

---

### 4.5 Synthetic Label Generation

Since Al-Safwa Bank has no historical ATM closure data, utility labels are **synthesised** from seven interpretable signals. This approach — using domain-knowledge-derived labels to supervise a GNN — is well-established in applied graph learning.

| Signal | Formula | Weight | Rationale |
|---|---|---|---|
| **Geographic isolation** | `normalize(Nearest ATM distance)` | 25 % | Far ATMs are irreplaceable |
| **Graph isolation** | `1 − normalize(node degree)` | 20 % | Few network neighbours = harder to replace |
| **Transaction volume** | `normalize(Daily Transactions)` | 20 % | Busy ATMs serve more customers |
| **Customer dependency** | `normalize(Unique Customers/Day)` | 15 % | Unique users relying on this ATM |
| **Service richness** | `Service Score / 3` | 10 % | Full-service ATMs (NFC + deposit + CW) are harder to replace |
| **Population density** | `normalize(Population Density 500m)` | 5 % | More people in the area → higher demand |
| **Power resilience** | `normalize(Generator Backup Hours)` | 5 % | Generator-backed ATMs survive outages |

```
utility_i = 0.25·isolation_geo + 0.20·isolation_graph + 0.20·transactions
           + 0.15·customers + 0.10·service + 0.05·population + 0.05·power
```

All signals are MinMax-normalised to [0, 1]. The composite score is then re-normalised to [0, 1].

**The GNN then learns to predict these labels from node features + graph topology.** After training, the GNN-refined scores are richer than the rule-based labels — the GNN learns, for example, that a well-connected ATM with full services in a dense cluster has a *different* network-level importance than an isolated ATM with the same raw features.

---

### 4.6 Training Details

| Hyperparameter | Value |
|---|---|
| Optimiser | Adam (`lr = 0.01`, `weight_decay = 5×10⁻⁴`) |
| Loss function | Mean Squared Error (MSELoss) |
| Epochs | 300 (default, configurable 100–500) |
| Dropout | 0.3 |
| Batch | Full graph (all 45 nodes per step) |
| Device | CPU (CUDA if available) |
| Seed | 42 (reproducible) |
| Best weights | Saved on minimum training loss |

With only 45 nodes, the full graph fits in memory and training completes in < 5 seconds on CPU.

---

### 4.7 What the GNN Learns

The GNN learns a function `f: (X, A) → û` where:
- `X` is the 45 × 13 node feature matrix
- `A` is the normalised adjacency matrix
- `û` is a 45-dimensional vector of utility scores

By doing message passing over the graph edges, the GNN answers a richer question than any per-node rule-based formula:

> *"Given this ATM's features AND the features and connectivity of all ATMs it is connected to — how critical is this ATM to the network?"*

An ATM that looks average in isolation may score high if all its neighbours are low-service alternatives. An ATM that is busy may score lower if five equally-busy ATMs sit within 2 km.

---

## 5. Model 2 — Integer Linear Programme (ILP)

### 5.1 Why ILP?

Once the GNN produces utility scores, the ATM selection problem becomes a **Binary Integer Programme** — a class of problem that ILP solvers can solve **exactly** (not approximately) in seconds for 45 variables. We use **PuLP** with the **CBC** solver (open source, no licence required).

---

### 5.2 Mathematical Formulation

```
Decision variables:
  x_i ∈ {0, 1}    for i = 1, …, 45
  (x_i = 1: keep ATM i;  x_i = 0: remove)

Objective:
  Maximise  Σᵢ (û_i · x_i)          — maximise total GNN utility of selected ATMs

Subject to:
  C1:  Σᵢ x_i  ≤  k                 — budget (keep at most k ATMs)
  C2:  Σᵢ x_i  ≥  max(1, k − 2)    — stay close to target k
  C3:  x_i = 0  ∀ i ∈ Mall_ATMs    — [curfew scenarios] force malls off
  C4:  x_i = 1  ∀ i ∈ Isolated     — protect geographic dead zones
  C5:  Σᵢ∈R x_i ≥ 1  ∀ region R   — at least 1 ATM per governorate
  C6:  Σᵢ∈G x_i ≥ ⌈0.30·k⌉        — [generator scenario] ≥30 % have backup power
```

---

### 5.3 Constraints Explained

| Constraint | Type | Justification |
|---|---|---|
| **C1 — Budget** | Hard | Regulatory mandate: reduce to k = ⌊45 × rate⌋ ATMs |
| **C2 — Near budget** | Hard | Prevents the solver from under-selecting; keeps selection close to k |
| **C3 — Curfew** | Hard (conditional) | Mall branches close under government curfew; forced to 0 |
| **C4 — Geographic guard** | Hard | ATMs where nearest alternative > 15 km cannot be removed — doing so creates a population with zero ATM access |
| **C5 — Regional coverage** | Hard | Every governorate must retain ≥ 1 ATM (regulatory minimum) |
| **C6 — Generator power** | Hard (conditional) | In rolling-outage scenarios, ≥ 30 % of kept ATMs must have ≥ 4 h backup |

The ILP is **always feasible** for the scenarios defined (verified: even the hardest scenario — 25 % + full curfew + generator — leaves enough street-branch ATMs with generator power to satisfy all constraints).

---

## 6. Model 3 — Scenario Sensitivity Analysis

Four operational scenarios are evaluated simultaneously, each applying different ILP constraint configurations:

| Scenario | ATMs to Keep | Malls Forced Off | Generator Rule | Colour |
|---|---|---|---|---|
| **Normal 50 % Reduction** | 22–23 | ❌ | ❌ | Blue |
| **50 % + Curfew** | 22–23 | ✅ (7 malls) | ❌ | Orange |
| **Severe 25 % Reduction** | 11 | ❌ | ❌ | Red |
| **25 % + Full Curfew + Generator** | 11 | ✅ (7 malls) | ✅ ≥ 30 % | Purple |

**The 7 mall ATMs** at high curfew risk:  
City Mall · Jerusalem St (B5 Mall) · Istiklal Mall · Taj Mall (Abdoun) · Mecca Mall · Irbid City Centre · Bab Al-Madina (Zarqa)

For each scenario, the system computes:
- Which ATMs are retained
- Geographic coverage % (customers within `radius_km` of a kept ATM)
- Daily revenue retained (JOD) = Σ transactions × avg value for kept ATMs
- Unique customers served per day

---

## 7. Data Pipeline

The simulated dataset contains **45 ATMs × 28 features** across 3 sheets.

### Input features (normalised to [0,1] for GNN)

| Feature | Type | Role in GNN |
|---|---|---|
| Daily Transactions | Continuous | Node feature |
| Unique Customers/Day | Continuous | Node feature |
| Avg Transaction Value (JOD) | Continuous | Node feature |
| Service Score 0–3 | Ordinal | Node feature (÷ 3) |
| Population Density (500 m) | Continuous | Node feature |
| Generator Backup Hours | Continuous | Node feature + C6 constraint |
| Is Mall | Binary | Node feature + C3 constraint |
| Is Generator | Binary | Node feature |
| Is Hybrid | Binary | Node feature |
| Has NFC | Binary | Node feature |
| Has Cash Deposit | Binary | Node feature |
| Has Contactless Withdrawal | Binary | Node feature |
| High Curfew Risk (mall = HIGH) | Binary | Node feature + C3 |

**Geospatial features** (Latitude, Longitude) are used to build the graph but are **not** fed as node features — the graph structure itself encodes spatial proximity.

---

## 8. Project Structure

```
Optimized ATM Network/
│
├── Sim_Data/
│   └── ATM_Simulated_Dataset_Safwa.xlsx    ← 45 ATMs × 28 features
│
├── src/                                    ← Core AI modules
│   ├── utils.py           Haversine distance, adjacency normalisation
│   ├── data_pipeline.py   Load · clean · encode · normalise data
│   ├── graph_builder.py   Build spatial proximity graph (NetworkX)
│   ├── gnn_model.py       GCN and GraphSAGE in pure PyTorch
│   ├── gnn_trainer.py     Synthetic labels · training loop
│   ├── ilp_optimizer.py   PuLP ILP with 5 constraint types
│   └── scenario_runner.py Run all 4 scenarios · compute metrics
│
├── dashboard/
│   └── app.py             Streamlit dashboard (Arabic RTL, Folium map)
│
├── outputs/
│   ├── scenario_results.json   ILP results per scenario
│   └── utility_scores.json     GNN scores per ATM
│
├── run.py                 CLI entry point
├── requirements.txt       Python dependencies
└── README.md              This file
```

---

## 9. How to Run

### Installation

```powershell
pip install -r requirements.txt
```

**Requirements:** Python ≥ 3.10 · PyTorch ≥ 2.0 (CPU is sufficient)

### Run the AI pipeline (CLI)

```powershell
# Default: GCN, 300 epochs, 5 km radius, all 4 scenarios
python run.py

# Use GraphSAGE instead of GCN
python run.py --arch sage

# Adjust training / graph parameters
python run.py --arch gcn --epochs 500 --radius 3.0
```

### Launch the interactive dashboard

```powershell
python -m streamlit run dashboard/app.py
```

Then open **http://localhost:8501** in your browser.

---

## 10. Dashboard Guide

| Section | Description |
|---|---|
| **Sidebar — Scenario selector** | Switch between 4 scenarios in real time |
| **Sidebar — Coverage radius** | Adjust the graph edge radius (2–15 km) |
| **Sidebar — GNN architecture** | Toggle GCN ↔ GraphSAGE |
| **Sidebar — Epochs** | Control training depth (faster vs. more accurate) |
| **KPI Cards** | ATMs kept · Coverage % · Revenue retained · Customers served |
| **Interactive Folium Map** | 🟢 Kept (street) · 🟡 Kept (mall) · 🔴 Removed; coverage circles; popup details |
| **GNN Utility Chart** | All 45 ATMs ranked by GNN score (green = kept, red = removed) |
| **Scenario Comparison** | Side-by-side bar charts across all 4 scenarios |
| **ATM Results Table** | Sortable table with all 28 features + GNN score progress bar |
| **Graph Insights** | Isolated node count · edge count · mall ATMs kept |
| **Export** | Download CSV or JSON for reporting |

---

## 11. Outputs & Metrics

### `outputs/utility_scores.json`
```json
{
  "Aqaba": 0.9821,
  "Karak": 0.9654,
  "Mafraq": 0.9412,
  ...
  "City Mall": 0.2134
}
```

### `outputs/scenario_results.json`
```json
{
  "normal_50": {
    "scenario": "Normal 50% Reduction",
    "atms_kept": 22,
    "coverage_pct": 92.5,
    "revenue_jod_day": 753762,
    "customers_served": 4602,
    "kept_atms": ["Shmeisani", "Wihdat", ...]
  },
  ...
}
```

### Metric definitions

| Metric | Formula |
|---|---|
| **Coverage %** | Customers within `radius_km` of any kept ATM ÷ total customers × 100 |
| **Revenue retained** | Σ (daily_transactions × avg_transaction_value) for kept ATMs |
| **Customers served** | Σ unique_customers/day for kept ATMs |

---

## 12. References

| Paper / Resource | Used For |
|---|---|
| Kipf & Welling (2017). *Semi-Supervised Classification with Graph Convolutional Networks*. ICLR. | GCN architecture |
| Hamilton, Ying & Leskovec (2017). *Inductive Representation Learning on Large Graphs*. NeurIPS. | GraphSAGE architecture |
| Wolsey (1998). *Integer Programming*. Wiley. | ILP formulation theory |
| Mitchell (2011). *PuLP: A Linear Programming Toolkit for Python*. | ILP solver |
| Haversine Formula — *Movable Type Scripts* | ATM pairwise distance computation |
| Folium Documentation — https://python-visualization.github.io/folium/ | Interactive ATM map |

---

*Al-Safwa Bank · Risk Management Department*  
*ATM Network Optimisation System — developed as part of the internship programme at Al-Hussein bin Abdullah Technical University (HTU)*
