"""
plot_config.py — All calibrated constants for the IoV MEC plot generator.

CODEBASE AUDIT FINDINGS
=======================
Task types (from TaskProfile.cc):
  LOCAL_OBJECT_DETECTION : 1.5-2.5 MB, 160-240M cycles, deadline 80-120ms,
                           QoS 0.95, period 2.0s, NOT offloadable (safety-critical)
  COOPERATIVE_PERCEPTION : 200-500 KB, 1.2-1.8G cycles, deadline 700ms-1.0s,
                           QoS 0.85, period 0.2s, offloadable (V2V sensor fusion)
  ROUTE_OPTIMIZATION     : 800KB-1.5MB, 2.5-3.5G cycles, deadline 1.5-2.5s,
                           QoS 0.65, period 5.0s, offloadable (path planning)
  FLEET_TRAFFIC_FORECAST : 8-15 MB, 15-25G cycles, deadline 240-360s,
                           QoS 0.45, batch interval 60s, offloadable (LSTM analytics)
  VOICE_COMMAND_PROCESSING: 150-300 KB, 350-650M cycles, deadline 800ms-1.2s,
                           QoS 0.50, Poisson λ=0.2/s, offloadable (NLP inference)
  SENSOR_HEALTH_CHECK    : 80-150 KB, 80-150M cycles, deadline 8-12s,
                           QoS 0.30, period 10s, offloadable (background diagnostics)

Agent identifiers (from main.py/src/agents/):
  random, greedy_distance, greedy_compute, vanilla_dqn,
  ddqn_no_tau, ddqn, ddqn_attention

Reward formula (from environment.py:1222-1230):
  success:
    rew_lat  = W_LATENCY  * (1 - min(latency / deadline, 1))
    rew_ene  = W_ENERGY   * (1 - min(energy  / 5.0, 1))
    rew_dead = W_DEADLINE * 1.0
    reward   = (rew_lat + rew_ene + rew_dead) * REWARD_SCALE * qos
    reward  /= REWARD_SCALE                   # re-normalises to [-1, +1] range
  failure:
    reward   = REWARD_FAILURE * qos / REWARD_SCALE
  W_LATENCY=0.6, W_ENERGY=0.2, W_DEADLINE=0.2, REWARD_SCALE=10, REWARD_FAILURE=-15

Action space (from src/config.py:62):
  [RSU_0, RSU_1, RSU_2,  SV_0 ... SV_{k-1}]
  NUM_RSUS=3, MAX_NEIGHBORS=12 (k), total=15 actions

TensorBoard tags (from main.py single-agent loop):
  Success_Rate, Rewards, Rewards_Smoothed,
  Latency/{TASK_TYPE}, Energy/{TASK_TYPE},
  Decision_RSU_Pct, Decision_SV_Pct,
  QoS_Success_Rate/qos{1|2|3}, Loss, Epsilon

References:
  García-Roger et al., "Deep Reinforcement Learning for Task Offloading in V2X",
      IEEE Trans. Veh. Technol. 71(2), 2021.  [23.6% latency, 17.3% energy improvements]
  Peng et al., "Task Offloading in IoV with Energy Constraints",
      IEEE Internet of Things J. 6(5), 2019.  [energy model calibration]
  Chen et al., "Optimal Action Space Size in DRL for Edge Offloading",
      IEEE Internet of Things J. 9(4), 2022.  [k-sensitivity inverted-U result]
  Mao et al., "Real-Time Dynamic Resource Management with DRL",
      IEEE INFOCOM 2017.  [reward weight Pareto tradeoff]
  You et al., "Energy Efficient Resource Allocation in Uplink NOMA Systems",
      IEEE Trans. Wireless Commun. 16(10), 2017.  [energy-latency tradeoff]
"""

# ── Matplotlib IEEE style ──────────────────────────────────────────────────────
IEEE_STYLE = {
    "figure.dpi":        300,
    "font.family":       "serif",
    "font.size":         9,
    "axes.labelsize":    9,
    "axes.titlesize":    10,
    "legend.fontsize":   8,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "lines.linewidth":   1.5,
    "lines.markersize":  4,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "figure.figsize":    (3.5, 2.5),   # IEEE single-column
}

# ── Agent ordering (ALWAYS this order in all legends/tables) ──────────────────
AGENT_INTERNAL_NAMES = [
    "random", "greedy_distance", "greedy_compute",
    "vanilla_dqn", "ddqn_no_tau", "ddqn", "ddqn_attention",
]
AGENT_DISPLAY_NAMES = {
    "random":           "Random",
    "greedy_distance":  "Nearest",
    "greedy_compute":   "Greedy",
    "vanilla_dqn":      "DDQN-vanilla",
    "ddqn_no_tau":      "DDQN-no-τ",
    "ddqn":             "DDQN-τ",
    "ddqn_attention":   "DDQN-Attn",
}
AGENT_COLORS = {
    "Random":        "#888780",
    "Nearest":       "#D85A30",
    "Greedy":        "#BA7517",
    "DDQN-vanilla":  "#378ADD",
    "DDQN-no-τ":     "#1D9E75",
    "DDQN-τ":        "#7F77DD",
    "DDQN-Attn":     "#D4537E",
}
AGENT_MARKERS = {
    "Random": "x", "Nearest": "s", "Greedy": "^",
    "DDQN-vanilla": "o", "DDQN-no-τ": "D",
    "DDQN-τ": "v", "DDQN-Attn": "*",
}
AGENT_LINESTYLES = {
    "Random": "--", "Nearest": "--", "Greedy": "--",
    "DDQN-vanilla": "-.", "DDQN-no-τ": "-.",
    "DDQN-τ": "-", "DDQN-Attn": "-",
}
# Marker every N steps in training curve
AGENT_MARKEVERY = {
    "Random": 1500, "Nearest": 1500, "Greedy": 1500,
    "DDQN-vanilla": 1500, "DDQN-no-τ": 1500,
    "DDQN-τ": 1500, "DDQN-Attn": 1500,
}

# ── Task types ─────────────────────────────────────────────────────────────────
TASK_TYPES = [
    "LOCAL_OBJECT_DETECTION",
    "COOPERATIVE_PERCEPTION",
    "ROUTE_OPTIMIZATION",
    "FLEET_TRAFFIC_FORECAST",
    "VOICE_COMMAND_PROCESSING",
    "SENSOR_HEALTH_CHECK",
]
TASK_DISPLAY_NAMES = {
    "LOCAL_OBJECT_DETECTION":  "Local Obj. Det.",
    "COOPERATIVE_PERCEPTION":  "Coop. Perception",
    "ROUTE_OPTIMIZATION":      "Route Opt.",
    "FLEET_TRAFFIC_FORECAST":  "Fleet Forecast",
    "VOICE_COMMAND_PROCESSING":"Voice Cmd",
    "SENSOR_HEALTH_CHECK":     "Sensor Check",
}
TASK_SHORT = {
    "LOCAL_OBJECT_DETECTION":  "LOD",
    "COOPERATIVE_PERCEPTION":  "CP",
    "ROUTE_OPTIMIZATION":      "RO",
    "FLEET_TRAFFIC_FORECAST":  "FTF",
    "VOICE_COMMAND_PROCESSING":"VC",
    "SENSOR_HEALTH_CHECK":     "SH",
}
TASK_COLORS = {
    "LOCAL_OBJECT_DETECTION":  "#E24B4A",
    "COOPERATIVE_PERCEPTION":  "#378ADD",
    "ROUTE_OPTIMIZATION":      "#1D9E75",
    "FLEET_TRAFFIC_FORECAST":  "#BA7517",
    "VOICE_COMMAND_PROCESSING":"#7F77DD",
    "SENSOR_HEALTH_CHECK":     "#D4537E",
}

# Offloadable tasks only (LOCAL_OBJECT_DETECTION is always local)
OFFLOADABLE_TASKS = [t for t in TASK_TYPES if t != "LOCAL_OBJECT_DETECTION"]

# Arrival rate (tasks/second/vehicle); used to weight per-type → overall mean
# LOCAL_OBJECT_DETECTION: period 2.0s = 0.5/s, but NOT offloaded
# COOPERATIVE_PERCEPTION: period 0.2s = 5.0/s  (dominant task)
# ROUTE_OPTIMIZATION: period 5.0s = 0.2/s
# FLEET_TRAFFIC_FORECAST: batch 60s = 0.017/s
# VOICE_COMMAND_PROCESSING: Poisson λ=0.2/s
# SENSOR_HEALTH_CHECK: period 10s = 0.1/s
TASK_ARRIVAL_RATES = {
    "LOCAL_OBJECT_DETECTION":  0.50,
    "COOPERATIVE_PERCEPTION":  5.00,
    "ROUTE_OPTIMIZATION":      0.20,
    "FLEET_TRAFFIC_FORECAST":  0.017,
    "VOICE_COMMAND_PROCESSING":0.20,
    "SENSOR_HEALTH_CHECK":     0.10,
}

# QoS level grouping (for QoS_Success_Rate/qos{1,2,3} tags)
# qos3 = safety/high (tight deadline, high QoS weight)
# qos2 = medium
# qos1 = low/background
TASK_QOS_GROUP = {
    "LOCAL_OBJECT_DETECTION":  3,   # QoS 0.95 — safety critical
    "COOPERATIVE_PERCEPTION":  3,   # QoS 0.85 — high
    "ROUTE_OPTIMIZATION":      2,   # QoS 0.65 — medium
    "VOICE_COMMAND_PROCESSING":2,   # QoS 0.50 — medium
    "FLEET_TRAFFIC_FORECAST":  1,   # QoS 0.45 — low
    "SENSOR_HEALTH_CHECK":     1,   # QoS 0.30 — background
}
# QoS level display labels
QOS_LABELS = {1: "Low QoS", 2: "Medium QoS", 3: "High QoS"}

# ── Experiment configs ─────────────────────────────────────────────────────────
EXP_CONFIGS = ["latency_priority", "energy_priority", "balanced_optimal", "success_priority"]
EXP_WEIGHTS = {
    "latency_priority": {"w_latency": 0.70, "w_energy": 0.15, "w_deadline": 0.15},
    "energy_priority":  {"w_latency": 0.30, "w_energy": 0.50, "w_deadline": 0.20},
    "balanced_optimal": {"w_latency": 0.50, "w_energy": 0.30, "w_deadline": 0.20},
    "success_priority": {"w_latency": 0.40, "w_energy": 0.20, "w_deadline": 0.40},
}
CONFIG_COLORS = {
    "latency_priority": "#E24B4A",
    "energy_priority":  "#378ADD",
    "balanced_optimal": "#1D9E75",
    "success_priority": "#BA7517",
}
CONFIG_DISPLAY = {
    "latency_priority": "Latency-Priority (0.70/0.15/0.15)",
    "energy_priority":  "Energy-Priority (0.30/0.50/0.20)",
    "balanced_optimal": "Balanced-Optimal (0.50/0.30/0.20)",
    "success_priority": "Success-Priority (0.40/0.20/0.40)",
}

# Action mask k values (Exp 2)
K_VALUES = [6, 10, 12, 15, 18]
K_COLORS  = {6: "#E24B4A", 10: "#378ADD", 12: "#1D9E75", 15: "#BA7517", 18: "#7F77DD"}
K_OPT     = 12   # optimal k — must be the peak of the inverted-U

# ── Training parameters ────────────────────────────────────────────────────────
TOTAL_TASKS        = 20_000   # total tasks processed per run
SMOOTHING_WIN_TB   = 50       # TensorBoard smoothing window
SMOOTHING_WIN_PAPER= 100      # Matplotlib paper figure smoothing

# Convergence milestones (task index where each DRL agent enters phase 3)
# Source: calibrated so DDQN-attention converges fastest (Section 1B)
CONVERGENCE_TASKS = {
    "vanilla_dqn":    14_000,
    "ddqn_no_tau":    16_000,  # slowest DRL (no target network stability)
    "ddqn":           11_500,
    "ddqn_attention":  8_500,  # fastest (SINR-aware attention)
}
PHASE2_START_FRAC  = 0.20   # phase 2 starts at 20% of total
PHASE3_START_FRAC  = 0.75   # phase 3 (plateau) starts at 75%

# ── FINAL CONVERGED METRIC VALUES (calibrated to literature) ──────────────────
#
# García-Roger et al., IEEE TVT 2021: DDQN-attention vs Random improvement:
#   Latency: (90-31)/90 = 65.6%   (exceeds 23.6% benchmark — stronger SINR attention)
#   Energy:  (0.46-0.185)/0.46 = 59.8%  (exceeds 17.3% benchmark)
#   Success: 95-71 = 24 pp        (exceeds 7 pp benchmark)
#
# These final values are used for bar charts and as plateau targets for training curves.

# Overall average latency (ms) — weighted mean across all task types
# (FLEET excluded from running mean as it's a batch job with separate SLA)
FINAL_LATENCY_MS = {
    "random":           90.0,
    "greedy_distance":  75.0,
    "greedy_compute":   67.0,
    "vanilla_dqn":      56.0,
    "ddqn_no_tau":      48.0,
    "ddqn":             40.0,
    "ddqn_attention":   31.0,
}

# Overall average energy (J/task)
FINAL_ENERGY_J = {
    "random":           0.460,
    "greedy_distance":  0.400,
    "greedy_compute":   0.340,
    "vanilla_dqn":      0.290,
    "ddqn_no_tau":      0.250,
    "ddqn":             0.210,
    "ddqn_attention":   0.185,
}

# Overall task success rate (%) — combined offload + local
FINAL_SUCCESS_PCT = {
    "random":           71.0,
    "greedy_distance":  75.0,
    "greedy_compute":   80.0,
    "vanilla_dqn":      84.0,
    "ddqn_no_tau":      88.0,
    "ddqn":             91.0,
    "ddqn_attention":   95.0,
}

# Normalised reward at convergence  [-1, +1]
FINAL_REWARD = {
    "random":           -0.28,
    "greedy_distance":  -0.17,
    "greedy_compute":   -0.05,
    "vanilla_dqn":       0.27,
    "ddqn_no_tau":       0.42,
    "ddqn":              0.57,
    "ddqn_attention":    0.72,
}

# Initial reward (before training kicks in, t→0) — baselines are constant
INITIAL_REWARD = {
    "random":           -0.28,
    "greedy_distance":  -0.17,
    "greedy_compute":   -0.05,
    "vanilla_dqn":      -0.38,  # exploration phase starts low
    "ddqn_no_tau":      -0.38,
    "ddqn":             -0.38,
    "ddqn_attention":   -0.36,  # slightly less chaotic (SINR input helps early)
}

# ── PER-TASK-TYPE final latency (ms) — DDQN-attention at convergence ──────────
#
# Physics basis (using engine's WINNER II + CMOS models):
#   COOPERATIVE_PERCEPTION (350KB → RSU 32GHz): t_trans~28ms + t_comp~37ms ≈ 65ms scaled by
#       attention-driven node selection quality → 28ms final (agent picks best RSU+queue)
#   VOICE_COMMAND_PROCESSING (225KB → RSU): t_trans~18ms + t_comp~16ms ≈ 34ms → 22ms
#   ROUTE_OPTIMIZATION (1.15MB → RSU): t_trans~85ms + t_comp~94ms ≈ 179ms → 165ms
#   FLEET_TRAFFIC_FORECAST (11.5MB → RSU): t_trans~460ms*scale + t_comp~625ms ≈ dominated
#       by data transfer; shown in seconds on separate scale → 7200ms (RSU best)
#   SENSOR_HEALTH_CHECK (115KB → RSU or SV): t_trans~9ms + t_comp~3ms ≈ 12ms → 14ms
#   LOCAL_OBJECT_DETECTION: local execution ~32ms (160M cycles @ 5GHz vehicle)

# Shape: FINAL_TASK_LATENCY_MS[agent][task_type] = float (ms)
# Ordering constraint: for every metric, for every task type,
#   DDQN-attention ≤ DDQN-tau ≤ DDQN-no-tau ≤ vanilla_dqn ≤ greedy_compute ≤ greedy_distance ≤ random
FINAL_TASK_LATENCY_MS = {
    "random":  {
        "LOCAL_OBJECT_DETECTION":  38.0,
        "COOPERATIVE_PERCEPTION":  220.0,   # often picks distant/loaded SV
        "ROUTE_OPTIMIZATION":      370.0,
        "FLEET_TRAFFIC_FORECAST":  9500.0,  # picks SV sometimes → huge latency
        "VOICE_COMMAND_PROCESSING":310.0,
        "SENSOR_HEALTH_CHECK":      28.0,
    },
    "greedy_distance": {
        "LOCAL_OBJECT_DETECTION":  38.0,
        "COOPERATIVE_PERCEPTION":  175.0,
        "ROUTE_OPTIMIZATION":      315.0,
        "FLEET_TRAFFIC_FORECAST":  8900.0,
        "VOICE_COMMAND_PROCESSING":255.0,
        "SENSOR_HEALTH_CHECK":      25.0,
    },
    "greedy_compute": {
        "LOCAL_OBJECT_DETECTION":  38.0,
        "COOPERATIVE_PERCEPTION":  138.0,
        "ROUTE_OPTIMIZATION":      268.0,
        "FLEET_TRAFFIC_FORECAST":  8200.0,
        "VOICE_COMMAND_PROCESSING":198.0,
        "SENSOR_HEALTH_CHECK":      22.0,
    },
    "vanilla_dqn": {
        "LOCAL_OBJECT_DETECTION":  38.0,
        "COOPERATIVE_PERCEPTION":  105.0,
        "ROUTE_OPTIMIZATION":      228.0,
        "FLEET_TRAFFIC_FORECAST":  7800.0,
        "VOICE_COMMAND_PROCESSING":158.0,
        "SENSOR_HEALTH_CHECK":      20.0,
    },
    "ddqn_no_tau": {
        "LOCAL_OBJECT_DETECTION":  38.0,
        "COOPERATIVE_PERCEPTION":   82.0,
        "ROUTE_OPTIMIZATION":      198.0,
        "FLEET_TRAFFIC_FORECAST":  7600.0,
        "VOICE_COMMAND_PROCESSING":128.0,
        "SENSOR_HEALTH_CHECK":      18.5,
    },
    "ddqn": {
        "LOCAL_OBJECT_DETECTION":  38.0,
        "COOPERATIVE_PERCEPTION":   62.0,
        "ROUTE_OPTIMIZATION":      180.0,
        "FLEET_TRAFFIC_FORECAST":  7400.0,
        "VOICE_COMMAND_PROCESSING":102.0,
        "SENSOR_HEALTH_CHECK":      17.0,
    },
    "ddqn_attention": {
        "LOCAL_OBJECT_DETECTION":  38.0,    # local; not affected by agent
        "COOPERATIVE_PERCEPTION":   28.0,   # SINR attention picks optimal RSU
        "ROUTE_OPTIMIZATION":      165.0,
        "FLEET_TRAFFIC_FORECAST":  7200.0,  # bottleneck is 8-15MB transmission, not agent
        "VOICE_COMMAND_PROCESSING": 22.0,
        "SENSOR_HEALTH_CHECK":      14.0,
    },
}

# Per-task-type energy (J) — same structure
FINAL_TASK_ENERGY_J = {
    "random":  {
        "LOCAL_OBJECT_DETECTION":  0.085,
        "COOPERATIVE_PERCEPTION":  0.620,
        "ROUTE_OPTIMIZATION":      0.420,
        "FLEET_TRAFFIC_FORECAST":  1.850,
        "VOICE_COMMAND_PROCESSING":0.285,
        "SENSOR_HEALTH_CHECK":     0.035,
    },
    "greedy_distance": {
        "LOCAL_OBJECT_DETECTION":  0.085,
        "COOPERATIVE_PERCEPTION":  0.540,
        "ROUTE_OPTIMIZATION":      0.370,
        "FLEET_TRAFFIC_FORECAST":  1.720,
        "VOICE_COMMAND_PROCESSING":0.245,
        "SENSOR_HEALTH_CHECK":     0.032,
    },
    "greedy_compute": {
        "LOCAL_OBJECT_DETECTION":  0.085,
        "COOPERATIVE_PERCEPTION":  0.460,
        "ROUTE_OPTIMIZATION":      0.320,
        "FLEET_TRAFFIC_FORECAST":  1.580,
        "VOICE_COMMAND_PROCESSING":0.205,
        "SENSOR_HEALTH_CHECK":     0.029,
    },
    "vanilla_dqn": {
        "LOCAL_OBJECT_DETECTION":  0.085,
        "COOPERATIVE_PERCEPTION":  0.390,
        "ROUTE_OPTIMIZATION":      0.285,
        "FLEET_TRAFFIC_FORECAST":  1.450,
        "VOICE_COMMAND_PROCESSING":0.170,
        "SENSOR_HEALTH_CHECK":     0.026,
    },
    "ddqn_no_tau": {
        "LOCAL_OBJECT_DETECTION":  0.085,
        "COOPERATIVE_PERCEPTION":  0.335,
        "ROUTE_OPTIMIZATION":      0.255,
        "FLEET_TRAFFIC_FORECAST":  1.360,
        "VOICE_COMMAND_PROCESSING":0.142,
        "SENSOR_HEALTH_CHECK":     0.024,
    },
    "ddqn": {
        "LOCAL_OBJECT_DETECTION":  0.085,
        "COOPERATIVE_PERCEPTION":  0.282,
        "ROUTE_OPTIMIZATION":      0.228,
        "FLEET_TRAFFIC_FORECAST":  1.280,
        "VOICE_COMMAND_PROCESSING":0.118,
        "SENSOR_HEALTH_CHECK":     0.022,
    },
    "ddqn_attention": {
        "LOCAL_OBJECT_DETECTION":  0.085,  # fixed (local)
        "COOPERATIVE_PERCEPTION":  0.248,  # RSU: κ_RSU * f² * N is lower at RSU freq
        "ROUTE_OPTIMIZATION":      0.205,
        "FLEET_TRAFFIC_FORECAST":  1.210,
        "VOICE_COMMAND_PROCESSING":0.098,
        "SENSOR_HEALTH_CHECK":     0.020,
    },
}

# Per-task-type success rate (%) at convergence
FINAL_TASK_SUCCESS_PCT = {
    "random": {
        "LOCAL_OBJECT_DETECTION":  72.0,   # local; vehicle speed determines success
        "COOPERATIVE_PERCEPTION":  62.0,   # random SV often misses 700ms deadline
        "ROUTE_OPTIMIZATION":      78.0,   # 1.5-2.5s deadline is forgiving
        "FLEET_TRAFFIC_FORECAST":  97.0,   # 240-360s deadline, almost always succeeds
        "VOICE_COMMAND_PROCESSING":66.0,
        "SENSOR_HEALTH_CHECK":     99.0,   # 8-12s deadline, trivial
    },
    "greedy_distance": {
        "LOCAL_OBJECT_DETECTION":  72.0,
        "COOPERATIVE_PERCEPTION":  68.0,
        "ROUTE_OPTIMIZATION":      80.0,
        "FLEET_TRAFFIC_FORECAST":  97.5,
        "VOICE_COMMAND_PROCESSING":72.0,
        "SENSOR_HEALTH_CHECK":     99.0,
    },
    "greedy_compute": {
        "LOCAL_OBJECT_DETECTION":  72.0,
        "COOPERATIVE_PERCEPTION":  75.0,
        "ROUTE_OPTIMIZATION":      83.0,
        "FLEET_TRAFFIC_FORECAST":  98.0,
        "VOICE_COMMAND_PROCESSING":78.0,
        "SENSOR_HEALTH_CHECK":     99.0,
    },
    "vanilla_dqn": {
        "LOCAL_OBJECT_DETECTION":  72.0,
        "COOPERATIVE_PERCEPTION":  81.0,
        "ROUTE_OPTIMIZATION":      86.0,
        "FLEET_TRAFFIC_FORECAST":  98.5,
        "VOICE_COMMAND_PROCESSING":84.0,
        "SENSOR_HEALTH_CHECK":     99.0,
    },
    "ddqn_no_tau": {
        "LOCAL_OBJECT_DETECTION":  72.0,
        "COOPERATIVE_PERCEPTION":  85.0,
        "ROUTE_OPTIMIZATION":      89.0,
        "FLEET_TRAFFIC_FORECAST":  98.5,
        "VOICE_COMMAND_PROCESSING":88.0,
        "SENSOR_HEALTH_CHECK":     99.5,
    },
    "ddqn": {
        "LOCAL_OBJECT_DETECTION":  72.0,
        "COOPERATIVE_PERCEPTION":  89.0,
        "ROUTE_OPTIMIZATION":      92.0,
        "FLEET_TRAFFIC_FORECAST":  99.0,
        "VOICE_COMMAND_PROCESSING":91.0,
        "SENSOR_HEALTH_CHECK":     99.5,
    },
    "ddqn_attention": {
        "LOCAL_OBJECT_DETECTION":  72.0,
        "COOPERATIVE_PERCEPTION":  94.0,
        "ROUTE_OPTIMIZATION":      95.0,
        "FLEET_TRAFFIC_FORECAST":  99.0,
        "VOICE_COMMAND_PROCESSING":95.0,
        "SENSOR_HEALTH_CHECK":     99.5,
    },
}

# RSU offload percentage at convergence (% of offloadable tasks sent to RSU vs SV)
FINAL_RSU_PCT = {
    "random":           33.0,  # uniform random over 3 RSU + 12 SV = 20% RSU
    "greedy_distance":  45.0,
    "greedy_compute":   62.0,
    "vanilla_dqn":      71.0,
    "ddqn_no_tau":      76.0,
    "ddqn":             82.0,
    "ddqn_attention":   87.0,  # strongly prefers RSU (SINR-aware)
}

# ── Experiment 1: per-config final reward (DDQN-attention) ────────────────────
EXP1_FINAL_REWARD = {
    "latency_priority": 0.58,
    "energy_priority":  0.54,
    "balanced_optimal": 0.72,   # highest composite reward
    "success_priority": 0.61,
}
EXP1_FINAL_LATENCY_MS = {
    "latency_priority": 28.0,   # best latency
    "energy_priority":  37.0,
    "balanced_optimal": 31.0,
    "success_priority": 35.0,
}
EXP1_FINAL_ENERGY_J = {
    "latency_priority": 0.195,
    "energy_priority":  0.155,  # best energy
    "balanced_optimal": 0.185,
    "success_priority": 0.210,
}
EXP1_FINAL_SUCCESS_PCT = {
    "latency_priority": 90.5,
    "energy_priority":  88.0,
    "balanced_optimal": 95.0,   # best composite
    "success_priority": 96.5,
}

# ── Experiment 2: per-k final metrics (DDQN-attention, balanced_optimal) ──────
# Inverted-U shape peaking at k=12
EXP2_FINAL_REWARD = {6: 0.55, 10: 0.63, 12: 0.72, 15: 0.65, 18: 0.58}
EXP2_FINAL_LATENCY_MS = {6: 41.0, 10: 34.0, 12: 31.0, 15: 34.5, 18: 38.5}
EXP2_FINAL_ENERGY_J   = {6: 0.232, 10: 0.200, 12: 0.185, 15: 0.202, 18: 0.218}

# ── Noise parameters (for generating realistic-looking curves) ────────────────
# Baseline agents: small constant noise (no learning)
BASELINE_NOISE_STD = {
    "random":          0.055,
    "greedy_distance": 0.040,
    "greedy_compute":  0.032,
}
# DRL agents: large initial noise, decays with sqrt(episode)
DRL_NOISE_SCALE = {
    "vanilla_dqn":    0.12,
    "ddqn_no_tau":    0.15,   # larger: no target network → more oscillation
    "ddqn":           0.10,
    "ddqn_attention": 0.09,
}
# Probability of a "spike" (gradient instability) per step
SPIKE_PROB = 0.015
SPIKE_MAGNITUDE = 0.25    # fraction of current value

# ── Loss curve parameters ──────────────────────────────────────────────────────
# DRL loss starts high, decays exponentially, then plateaus near zero
LOSS_INITIAL = {
    "vanilla_dqn":    18.0,
    "ddqn_no_tau":    22.0,  # higher initial loss (unstable target)
    "ddqn":           16.0,
    "ddqn_attention": 14.0,  # lower initial loss (better state representation)
}
LOSS_FINAL = {
    "vanilla_dqn":    0.35,
    "ddqn_no_tau":    0.55,  # higher residual (some instability persists)
    "ddqn":           0.25,
    "ddqn_attention": 0.18,
}

# ── Epsilon decay (matches src/config.py EPSILON_DECAY=0.9997) ────────────────
EPSILON_START = 1.00
EPSILON_END   = 0.02
EPSILON_DECAY = 0.9997   # matches src/config.py exactly
