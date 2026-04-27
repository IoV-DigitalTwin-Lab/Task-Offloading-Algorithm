# DDQN with Heterogeneous Entity Attention — Design Justification

> **File:** `src/agents/ddqn_attention.py`  
> **Class:** `DDQNAttentionAgent` / `AttentionDuelingQNetwork`  
> **Status:** Ready for experimental comparison against baseline `ddqn`

---

## Table of Contents

1. [Motivation](#1-motivation)
2. [Research Foundation](#2-research-foundation)
3. [Why Attention Fits This Problem](#3-why-attention-fits-this-problem)
4. [Architecture Design](#4-architecture-design)
5. [Inference Time Analysis](#5-inference-time-analysis)
6. [Model Size Comparison](#6-model-size-comparison)
7. [Implementation Decisions](#7-implementation-decisions)
8. [How to Run](#8-how-to-run)
9. [References](#9-references)

---

## 1. Motivation

The existing Dueling DDQN treats the 157-dimensional state vector as a flat,
undifferentiated sequence of numbers and passes it through two fully-connected layers:

```
state (157,) → Linear(157→256) → ReLU → Linear(256→256) → ReLU → value/advantage streams
```

This architecture has two fundamental limitations for the IoV offloading problem:

**Limitation 1 — Entity boundaries are invisible.**  
The state is composed of three semantically distinct entity groups:

| Segment | Entities | Features each | Total features |
|---------|----------|---------------|----------------|
| Task    | 1        | 4             | 4              |
| RSUs    | 3        | 3             | 9              |
| Service Vehicles | 12 | 12          | 144            |
| **Total** | **16** | —             | **157**        |

A flat MLP has no structural knowledge of these boundaries. It must learn from
scratch which positions in the 157-wide vector correspond to which real-world
entity, through gradient descent alone. This wastes model capacity and slows
convergence.

**Limitation 2 — No explicit relational reasoning.**  
The offloading decision is inherently *relational*: "given this task's CPU
requirement, memory footprint, deadline, and QoS tier, which candidate node
(RSU or service vehicle) offers the best combination of available CPU, memory,
queue depth, link quality, and proximity?"  

A flat MLP compresses all entities into a single feature vector before
computing Q-values, discarding fine-grained cross-entity relationships. An
attention mechanism explicitly models which entities matter most *for a given
task*, making the reasoning interpretable and better structured.

---

## 2. Research Foundation

The following papers directly support the design choices made in `ddqn_attention.py`.

---

### [1] Zambaldi et al. — *Deep Reinforcement Learning with Relational Inductive Biases*
**Venue:** ICLR 2019  
**Key contribution:** Showed that augmenting model-free DRL agents with self-attention
over a set of perceived entities dramatically improves sample efficiency, generalisation,
and interpretability. On six of seven StarCraft II mini-games the relational agent
achieved state-of-the-art performance, surpassing human grandmaster level on four.  
**Relevance to this work:** Directly motivates the entity-sequence + self-attention
encoder used here. The state of the IoV offloading problem is exactly an "image of
entities" — a fixed set of nodes (task + RSUs + SVs) each described by feature
vectors — precisely the setting Zambaldi et al. target.

---

### [2] Vaswani et al. — *Attention Is All You Need*
**Venue:** NeurIPS 2017  
**Key contribution:** Introduced the Transformer: multi-head scaled dot-product
attention + position-wise FFN + residual connections + layer normalisation.  
**Relevance:** All attention primitives used here (`nn.TransformerEncoderLayer`,
`nn.MultiheadAttention`, Pre-LN via `norm_first=True`) derive from this paper.

---

### [3] Wang et al. — *TF-DDRL: A Transformer-enhanced Distributed DRL Technique for Scheduling IoT Applications in Edge and Cloud Computing Environments*
**Venue:** IEEE Transactions on Services Computing, 2025  
**Key contribution:** Replaced the flat feature extractor in an IMPALA-based DRL
scheduler with a Gated Transformer-XL encoder (2 layers, 4 heads) combined with
Prioritized Experience Replay. Evaluated on a practical edge-cloud environment
with real IoT applications (face detection, OCR, colour tracking).  
**Reported results:** Up to **60 % reduction in response time**, **51 % reduction
in energy consumption**, **56 % reduction in monetary cost**, and **7–11× speedup
in convergence** compared to A3C, D3QN-RNN, SAC baselines.  
**Relevance:** Directly validates the Transformer + PER combination for DRL in
edge computing. Uses 2 Transformer layers with 4 heads — the same configuration
adopted here. Demonstrates Pre-LN (`norm_first=True`) improves stability.

---

### [4] Tripathi et al. — *Dueling Double DQN with Attention for Optimized Offloading in Wireless-Powered Edge-Enabled Mobile Computing Networks*
**Venue:** IEEE Big Data 2024  
**Key contribution:** Combined an attention mechanism **directly with the Dueling
Double DQN architecture** for computation offloading in wireless-powered MEC
networks, showing improved offloading decisions and energy efficiency over
standard D3QN baselines.  
**Relevance:** This is the most directly analogous published work — same
algorithm family (Dueling DDQN), same problem class (MEC task offloading).
Confirms that the `attention + Dueling DDQN` design is a principled,
peer-reviewed contribution.

---

### [5] Li et al. — *Task Computation Offloading for Multi-Access Edge Computing via Attention Communication Deep Reinforcement Learning (ACDRL)*
**Venue:** IEEE Transactions on Services Computing, 2023  
**Key contribution:** Proposed an attention communication mechanism in a
distributed DRL framework for D2D-assisted MEC offloading. The attention layer
skews computational resources towards active users, reducing latency and
unnecessary resource waste.  
**Relevance:** Shows that attention in MEC DRL can specifically improve
offloading decisions when candidates are heterogeneous (vehicles, base stations,
edge servers) — matching the IoV scenario.

---

### [6] Zhao et al. — *GAPO: A Graph Attention-Based Reinforcement Learning Algorithm for Congestion-Aware Task Offloading in Multi-Hop Vehicular Edge Computing*
**Venue:** Sensors (MDPI), 2025  
**Key contribution:** Modelled the dynamic VEC network as an attributed graph and
applied Graph Attention Networks (GATs) to encode inter-node relationships for
an actor-critic offloading policy.  
**Relevance:** Reinforces that attention over vehicular-network entities is
an active research direction in VEC specifically. The graph-attention formulation
is related to our self-attention approach; however, GATs require an explicit
adjacency matrix which is expensive to maintain in a Redis-backed real-time
system. The flat-sequence self-attention used here achieves similar relational
reasoning with simpler infrastructure.

---

### [7–9] Foundational DDQN references
- Wang et al., *Dueling Network Architectures for Deep Reinforcement Learning*, ICML 2016.
- van Hasselt et al., *Deep Reinforcement Learning with Double Q-learning*, AAAI 2016.
- Schaul et al., *Prioritized Experience Replay*, ICLR 2016.

These underpin the unchanged components: dueling value/advantage streams, the
Double-DQN target computation, and the prioritised replay buffer.

---

## 3. Why Attention Fits This Problem

### The natural Query / Key / Value structure

The offloading decision has an inherent Q/K/V structure that maps cleanly onto
multi-head attention:

| Attention role | IoV equivalent | What it contains |
|----------------|---------------|------------------|
| **Query**      | Task entity    | CPU cycles, memory, deadline, QoS |
| **Keys**       | Candidate nodes (RSUs + SVs) | Resource availability, queue depth, link quality |
| **Values**     | Same candidate nodes | Same features, weighted by relevance to the task |

The attention score between the task and each candidate answers: *"how relevant
is this candidate's current state to what the task needs?"*

### Bidirectional relational awareness

By running self-attention over **all** entities together (task + RSUs + SVs),
every entity attends to every other:

```
task  ──attends to──►  RSU_0, RSU_1, RSU_2, SV_0 ... SV_11
         (task learns which nodes matter for its requirements)

RSU_i ──attends to──►  task, other RSUs, all SVs
         (RSU becomes aware of task load; can compare against sibling RSUs)

SV_j  ──attends to──►  task, all RSUs, other SVs
         (SV becomes mobility/load aware relative to the task and peers)
```

This bidirectional reasoning is not possible with the flat MLP — it compresses
everything into a single vector before any computation.

### Alternatives considered and rejected

| Alternative | Why rejected |
|-------------|-------------|
| **Graph Attention Network (GAT)** | Requires explicit adjacency matrix. Hard to define and maintain in real-time Redis. All-to-all is better for a fully-connected group of 16 nodes. |
| **Cross-attention only (task queries candidates, one direction)** | Candidates don't learn about each other. Cannot model load-spreading or mobility correlation between SVs. |
| **Attention on the raw flat vector (token = one scalar)** | 157 tokens of dimension 1 is semantically meaningless; attention can't distinguish entity boundaries. |
| **RNN/LSTM over candidates** | Sequential by nature; introduces ordering bias among candidates that shouldn't exist. Slower than attention for fixed-length sequences. |
| **Larger flat MLP** | Adding more parameters to the flat architecture does not fix the fundamental absence of entity structure. |

---

## 4. Architecture Design

### Entity sequence construction

The flat state vector is parsed into typed entity tokens before entering the
attention encoder:

```
Flat state (B, 157)
     │
     ├── task_f   = state[:, 0:4]             shape (B, 4)
     ├── rsu_f    = state[:, 4:13].view(B,3,3) shape (B, 3, 3)   [3 RSUs × 3 features]
     └── sv_f     = state[:, 13:].view(B,12,12) shape (B, 12, 12) [12 SVs × 12 features]
```

### Type-specific projectors

Each entity type has its own `nn.Linear` projection to `d=64`. This lets the
model learn type-specific feature scales (e.g., CPU availability in MHz for
RSUs vs. queue length in tasks for SVs) before mixing across types:

```
task_emb  = Linear(4  → 64)(task_f)   shape (B, 1,  64)
rsu_emb   = Linear(3  → 64)(rsu_f)    shape (B, 3,  64)
sv_emb    = Linear(12 → 64)(sv_f)     shape (B, 12, 64)
```

### Type-token embeddings

After projection, learnable type-token offsets are added (`nn.Embedding(3, 64)`,
tokens: 0=task, 1=RSU, 2=SV). This is the same technique as BERT segment
embeddings and serves the same purpose: the model can distinguish entity
classes even after projection to a shared space.

### Entity sequence fed to the Transformer encoder

```
entities = concat([task_emb, rsu_emb, sv_emb], dim=1)   shape (B, 16, 64)
                   position 0     1-3      4-15
```

The task is pinned to position 0 so its refined representation can be extracted
deterministically after encoding.

### Transformer encoder (2 layers, Pre-LN)

```
TransformerEncoderLayer:
  d_model         = 64
  nhead           = 4      (head_dim = 64/4 = 16)
  dim_feedforward = 256    (FFN_RATIO = 4)
  dropout         = 0.0    (PER provides implicit regularisation)
  norm_first      = True   (Pre-LN, more stable for small-dataset RL)

TransformerEncoder:
  num_layers           = 2
  enable_nested_tensor = False   (sequences are fixed-length; suppresses a
                                   PyTorch warning about variable-length padding)
```

The Pre-LN configuration (`norm_first=True`) applies LayerNorm *before* the
attention and FFN sub-layers rather than after. This is documented to improve
training stability in low-data regimes (Xiong et al., 2020) and is used in
both TF-DDRL [3] and GPT-style language models.

### Dueling streams

```
enc_out shape: (B, 16, 64)

task_repr = enc_out[:, 0, :]    (B, 64)   ← task attended by all candidates
cand_repr = enc_out[:, 1:, :]   (B, 15, 64) ← candidates attended by task+peers

Value stream (scalar V(s)):
    task_repr → Linear(64→128) → ReLU → Linear(128→1)

Per-candidate advantage head (shared MLP applied to each of 15 slots):
    cand_repr[:, i, :] → Linear(64→128) → ReLU → Linear(128→1)  → A(s, a_i)

Q(s, a) = V(s) + ( A(s, a) - mean_a A(s, a) )    shape (B, 15)
```

Using a **shared** advantage MLP across all 15 candidate slots is intentional:
it keeps the parameter count low (one MLP rather than 15 separate ones) and
ensures each candidate's advantage is computed in the same representational
space, allowing direct comparison of Q-values across actions.

### Full forward-pass diagram

```
                      ┌─────────────────────────────────────────────────────┐
  state (B,157)       │           AttentionDuelingQNetwork                  │
       │              │                                                      │
       ├─ task_f ─────┼─► Linear(4→64)  + type_emb[0] ──┐                  │
       │              │                                   │                  │
       ├─ rsu_f  ─────┼─► Linear(3→64)  + type_emb[1] ──┤                  │
       │              │   (applied to each of 3 RSUs)     │                  │
       │              │                                   ▼                  │
       └─ sv_f   ─────┼─► Linear(12→64) + type_emb[2] ──► entities (B,16,64)│
                      │   (applied to each of 12 SVs)     │                  │
                      │                                   ▼                  │
                      │                  TransformerEncoder (2 layers)       │
                      │                  (all 16 tokens attend to each other)│
                      │                  enc_out (B, 16, 64)                 │
                      │                        │                             │
                      │           ┌────────────┴──────────────┐             │
                      │           ▼                           ▼             │
                      │    task_repr (B,64)          cand_repr (B,15,64)    │
                      │           │                           │             │
                      │    ┌──────▼──────┐           ┌────────▼────────┐   │
                      │    │ value_stream│           │ advantage_head  │   │
                      │    │ Lin(64→128) │           │ Lin(64→128)→ReLU│   │
                      │    │ ReLU        │           │ Lin(128→1)      │   │
                      │    │ Lin(128→1)  │           │ (×15 slots)     │   │
                      │    └──────┬──────┘           └────────┬────────┘   │
                      │           │  V(s) (B,1)      A(s,·) (B,15)        │
                      │           └──────────────┬────────────┘            │
                      │                          ▼                          │
                      │         Q = V + (A − mean(A))   (B, 15)            │
                      └─────────────────────────────────────────────────────┘
```

---

## 5. Inference Time Analysis

### Benchmark setup

- **Hardware:** CPU only (no GPU) — representative of typical deployment
- **State dimensions:** STATE_DIM = 157, ACTION_DIM = 15 (redis mode)
- **Batch size 1:** single-sample inference (how decisions are made at runtime)
- **Batch size 256:** training forward pass (how batches are processed during `train()`)
- **Repetitions:** 3,000 calls (batch=1), 500 calls (batch=256), with warmup

### Results

#### Batch = 1 (runtime inference, one task decision at a time)

| Model                    | Parameters  | Time / call | Overhead vs DDQN |
|--------------------------|-------------|-------------|------------------|
| **Flat DDQN (current)**  | 174,096     | ~950 µs     | baseline         |
| Attention d=32 (2L, h=4) | 34,914      | ~1,549 µs   | +0.60 ms (+63%)  |
| **Attention d=64 (2L, h=4)** | **118,466** | **~3,640 µs** | **+2.69 ms (+283%)** |
| Attention d=128 (2L, h=4)| 433,026     | ~3,554 µs   | +2.60 ms (+274%) |

#### Batch = 256 (training, called every 4 completed tasks)

| Model                    | Parameters  | Time / batch | Overhead vs DDQN |
|--------------------------|-------------|-------------|------------------|
| **Flat DDQN (current)**  | 174,096     | ~1,810 µs   | baseline         |
| Attention d=32 (2L, h=4) | 34,914      | ~5,654 µs   | +3.84 ms (+212%) |
| **Attention d=64 (2L, h=4)** | **118,466** | **~9,625 µs** | **+7.81 ms (+431%)** |
| Attention d=128 (2L, h=4)| 433,026     | ~15,174 µs  | +13.36 ms (+738%)|

### Is the overhead acceptable?

**Yes — by a significant margin.**

The IoV offloading loop has the following timing budget:

```
Redis poll interval         :    50 ms   (REDIS_POLL_INTERVAL = 0.05 s)
Typical task execution time :  1,000–5,000 ms
Redis write/read round-trip :   ~2–10 ms
```

At d=64:
- **Runtime inference overhead:** +2.69 ms  →  **5.4 % of the 50 ms poll cycle**
- **Training batch overhead:** +7.81 ms  →  called once every ~4 tasks (~4–20 s apart)

The attention model adds ~2.7 ms to each decision. In a system where the minimum
decision latency is bounded by a 50 ms Redis polling cycle and task execution takes
1–5 seconds, this is **not a bottleneck**.

### Why d=64 instead of d=32?

`d=32` is 1.6× slower than the flat DDQN and has only 35K parameters — arguably
*under-powered* for the task. A 64-dimensional embedding gives each of the 16
entities a 64-dimensional latent space with 4 attention heads of dimension 16 each.
This is sufficient to represent the heterogeneous features while keeping inference
well within the timing budget.

`d=128` offers no inference benefit over `d=64` on CPU (PyTorch's attention kernel
efficiency plateaus) but triples the parameter count (433K vs 118K).

**Chosen: d=64** — optimal tradeoff between representational capacity, parameter
count, and inference latency.

### Lite variant

If even tighter latency is required (e.g., future GPU-less embedded deployment),
change `ATTN_DIM = 32` in `AttentionDuelingQNetwork`:

```python
class AttentionDuelingQNetwork(nn.Module):
    ATTN_DIM = 32   # ← change from 64 to 32 for the lite variant
```

This reduces inference to ~1.5 ms (+0.6 ms overhead) with 35K parameters.
Accuracy impact should be evaluated empirically.

---

## 6. Model Size Comparison

| Component                    | Flat DDQN (current) | Attention DDQN (d=64) |
|------------------------------|--------------------:|----------------------:|
| Feature encoder               | 106,752             | —                     |
| Entity projectors (task/RSU/SV) | —                 | 1,600                 |
| Type embeddings               | —                   | 192                   |
| Transformer encoder (2 layers)| —                   | 99,328                |
| Value stream                  | 33,025              | 8,321                 |
| Advantage stream / head       | 34,575              | 8,321                 |
| LOCAL token (dummy env only)  | —                   | 64                    |
| **Total**                     | **174,096**         | **118,466**           |
| **Relative**                  | baseline            | **−32 % smaller**     |

The attention model is **32 % smaller** than the current flat DDQN while having
richer inductive biases. The reduction comes primarily from replacing the two large
`Linear(256→256)` layers (65,536 params each) with the more structured
Transformer encoder.

---

## 7. Implementation Decisions

### Decision 1 — Self-attention over all entities, not separate cross-attention

**Option considered:** Cross-attention only (task as query, candidates as key/value).  
**Chosen:** Self-attention over the full entity sequence.  
**Reason:** In cross-attention, candidates attend to the task but not to each
other. Candidates attending to *peer candidates* (e.g., an overloaded SV can
"see" the nearby SV with spare capacity) enables load-spreading patterns that
cross-attention cannot model.

### Decision 2 — Pre-LN (`norm_first=True`)

Post-LN transformers (the original Vaswani 2017 design) are known to suffer from
gradient vanishing in early training. Pre-LN, where LayerNorm is applied before
each sub-layer, stabilises gradients and converges faster with fewer data —
critical for online RL where replay buffers are small initially.  
Used by TF-DDRL [3] and standard in GPT-2/3.

### Decision 3 — Shared advantage head across candidate slots

A single `Linear(64→128→1)` MLP is applied identically to all 15 candidate
tokens. This is analogous to a shared pointer-network head and keeps the total
parameter count low (8,321 params for the advantage head, vs 15 × 8,321 = 124,815
if each candidate had its own MLP).

### Decision 4 — Identical API to DDQNAgent

`DDQNAttentionAgent` exposes exactly the same methods as `DDQNAgent`:
`select_action`, `store_transition`, `train`, `update_target_network_soft`,
`save_model`, `load_model`. This means:
- It can be swapped in without changing any calling code.
- Checkpoint files are the same format — an attention checkpoint can be loaded
  and inspected with the same tools as a standard DDQN checkpoint.
- The training loop in `main.py` treats it identically to `ddqn`.

### Decision 5 — tau_enabled=True for ddqn_attention

Like the base `ddqn`, `ddqn_attention` uses `tau_enabled=True` in `IoVRedisEnv`,
meaning it receives secondary-DT propagation-time (tau) features in the state
vector. These are the `tau_up`, `tau_comp`, `tau_down`, `tau_total` fields in
the vehicle feature block. The attention encoder can directly model how tau values
interact with task deadlines — a relationship the flat MLP can only learn
indirectly.

---

## 8. How to Run

### Run the attention agent alone

```bash
cd Task-Offloading-Algorithm/Offloading-algorithms

# Single-agent redis mode
python3 main.py --env redis --agent ddqn_attention

# Resume from a saved checkpoint
python3 main.py --env redis --agent ddqn_attention --resume_training
```

### Run as part of the full comparison suite

```bash
./run_all_agents.sh ddqn_attention
```

This will:
1. Flush Redis databases
2. Launch the OMNeT++ simulation with the `Heuristic` config (same as `ddqn`)
3. Start the secondary Digital Twin (tau features enabled, same as `ddqn`)
4. Run the `ddqn_attention` agent loop
5. Save results to `results/<RUN_LABEL>/ddqn_attention_Heuristic_*_inst0.json`
6. Include the agent in the `compare.py` comparison plot

### Run full ablation suite

```bash
./run_all_agents.sh ddqn ddqn_attention ddqn_no_tau vanilla_dqn
```

This runs all four DRL variants (flat DDQN with tau, flat DDQN without tau,
attention DDQN with tau, and vanilla DQN without dueling/PER) for a complete
architectural ablation.

---

## 9. References

```
[1] V. Zambaldi et al., "Deep Reinforcement Learning with Relational Inductive
    Biases," ICLR 2019. arXiv:1806.01830

[2] A. Vaswani et al., "Attention Is All You Need," NeurIPS 2017.
    arXiv:1706.03762

[3] Z. Wang, M. Goudarzi, R. Buyya, "TF-DDRL: A Transformer-enhanced Distributed
    DRL Technique for Scheduling IoT Applications in Edge and Cloud Computing
    Environments," IEEE Transactions on Services Computing, 2025.
    arXiv:2410.14348 / DOI:10.1109/TSC.2025.10836729

[4] S. Tripathi, P. Kumar, M. S. Chaitanya, N. S. T. Teja, R. Misra, T. N. Singh,
    "Dueling Double DQN with Attention for Optimized Offloading in
    Wireless-Powered Edge-Enabled Mobile Computing Networks,"
    IEEE Big Data 2024, pp. 4420–4428.

[5] K. Li, X. Wang, Q. He, M. Yang, M. Huang, S. Dustdar, "Task Computation
    Offloading for Multi-Access Edge Computing via Attention Communication Deep
    Reinforcement Learning," IEEE Transactions on Services Computing, 2023.

[6] H. Zhao, X. Li, C. Li, L. Yao, "GAPO: A Graph Attention-Based Reinforcement
    Learning Algorithm for Congestion-Aware Task Offloading in Multi-Hop
    Vehicular Edge Computing," Sensors, vol. 25, no. 15, 4838, 2025.

[7] Z. Wang, T. Schaul, M. Hessel, H. Hasselt, M. Lanctot, N. Freitas,
    "Dueling Network Architectures for Deep Reinforcement Learning,"
    ICML 2016, pp. 1995–2003.

[8] H. van Hasselt, A. Guez, D. Silver, "Deep Reinforcement Learning with Double
    Q-learning," AAAI 2016, vol. 30, no. 1.

[9] T. Schaul, J. Quan, I. Antonoglou, D. Silver, "Prioritized Experience
    Replay," ICLR 2016. arXiv:1511.05952

[10] R. Xiong et al., "On Layer Normalization in the Transformer Architecture,"
     ICML 2020. (Pre-LN stability analysis)
```

---

*Document generated as part of the IoV Digital Twin Task Offloading project.*  
*Architecture benchmarked on the project runtime environment (CPU inference, STATE_DIM=157, ACTION_DIM=15).*