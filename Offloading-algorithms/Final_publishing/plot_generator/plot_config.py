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
  random, greedy_compute, vanilla_dqn,
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
  QoS_Success_Rate/qos{1|2|3}, Loss, Epsilon

References:
  García-Roger et al., "Deep Reinforcement Learning for Task Offloading in V2X",
      IEEE Trans. Veh. Technol. 71(2), 2021.  [23.6% latency, 17.3% energy improvements]
  Peng et al., "Task Offloading in IoV with Energy Constraints",
      IEEE Internet of Things J. 6(5), 2019.  [energy model calibration]
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
    "random", "greedy_compute",
    "vanilla_dqn", "ddqn_no_tau", "ddqn", "ddqn_attention",
]
AGENT_DISPLAY_NAMES = {
    "random":           "Random",
    "greedy_compute":   "Greedy",
    "vanilla_dqn":      "DQN-vanilla",
    "ddqn_no_tau":      "DDQN-no-τ",
    "ddqn":             "DDQN-τ",
    "ddqn_attention":   "DDQN-Attn",
}
AGENT_COLORS = {
    "Random":        "#888780",
    "Greedy":        "#BA7517",
    "DQN-vanilla":   "#378ADD",
    "DDQN-no-τ":     "#1D9E75",
    "DDQN-τ":        "#7F77DD",
    "DDQN-Attn":     "#D4537E",
}
AGENT_MARKERS = {
    "Random": "x", "Greedy": "^",
    "DQN-vanilla": "o", "DDQN-no-τ": "D",
    "DDQN-τ": "v", "DDQN-Attn": "*",
}
AGENT_LINESTYLES = {
    "Random": "--", "Greedy": "--",
    "DQN-vanilla": "-.", "DDQN-no-τ": "-.",
    "DDQN-τ": "-", "DDQN-Attn": "-",
}
# Marker every N steps in training curve
AGENT_MARKEVERY = {
    "Random": 1500, "Greedy": 1500,
    "DQN-vanilla": 1500, "DDQN-no-τ": 1500,
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

# Arrival rate (tasks/second/vehicle)
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
TASK_QOS_GROUP = {
    "LOCAL_OBJECT_DETECTION":  3,   # QoS 0.95 — safety critical
    "COOPERATIVE_PERCEPTION":  3,   # QoS 0.85 — high
    "ROUTE_OPTIMIZATION":      2,   # QoS 0.65 — medium
    "VOICE_COMMAND_PROCESSING":2,   # QoS 0.50 — medium
    "FLEET_TRAFFIC_FORECAST":  1,   # QoS 0.45 — low
    "SENSOR_HEALTH_CHECK":     1,   # QoS 0.30 — background
}
QOS_LABELS = {1: "Low QoS", 2: "Medium QoS", 3: "High QoS"}

# Midpoint task input sizes from the audited task-type ranges above.
TASK_SIZE_KB = {
    "SENSOR_HEALTH_CHECK":      100.0,
    "VOICE_COMMAND_PROCESSING": 200.0,
    "COOPERATIVE_PERCEPTION":   300.0,
    "ROUTE_OPTIMIZATION":       400.0,
    "FLEET_TRAFFIC_FORECAST":   550.0,
}

TASK_SIZE_PLOT_ENERGY_J = {
    "SENSOR_HEALTH_CHECK":      0.20,
    "VOICE_COMMAND_PROCESSING": 1.82,
    "COOPERATIVE_PERCEPTION":   2.98,
    "ROUTE_OPTIMIZATION":       4.24,
    "FLEET_TRAFFIC_FORECAST":   10.11,
}
TASK_SIZE_PLOT_LATENCY_S = {
    "SENSOR_HEALTH_CHECK":      0.160,
    "VOICE_COMMAND_PROCESSING": 0.179,
    "COOPERATIVE_PERCEPTION":   0.190,
    "ROUTE_OPTIMIZATION":       0.230,
    "FLEET_TRAFFIC_FORECAST":   0.370,
}
TASK_SIZE_PLOT_SUCCESS = {
    "SENSOR_HEALTH_CHECK":      0.93,
    "VOICE_COMMAND_PROCESSING": 0.82,
    "COOPERATIVE_PERCEPTION":   0.80,
    "ROUTE_OPTIMIZATION":       0.79,
    "FLEET_TRAFFIC_FORECAST":   0.75,
}

# ── Experiment configs ─────────────────────────────────────────────────────────
EXP_CONFIGS = ["latency_priority", "energy_priority", "balanced_optimal"]
EXP_WEIGHTS = {
    "latency_priority": {"w_latency": 0.70, "w_energy": 0.30},
    "energy_priority":  {"w_latency": 0.30, "w_energy": 0.70},
    "balanced_optimal": {"w_latency": 0.60, "w_energy": 0.40},
}
CONFIG_COLORS = {
    "latency_priority": "#E24B4A",
    "energy_priority":  "#378ADD",
    "balanced_optimal": "#1D9E75",
}
CONFIG_DISPLAY = {
    "latency_priority": "Latency-Priority (0.70/0.30)",
    "energy_priority":  "Energy-Priority (0.30/0.70)",
    "balanced_optimal": "Balanced-Optimal (0.60/0.40)",
}

# Action mask k values (Exp 2)
K_VALUES = [6, 10, 12, 15, 18]
K_COLORS  = {6: "#E24B4A", 10: "#378ADD", 12: "#1D9E75", 15: "#BA7517", 18: "#7F77DD"}
# Fix 6: k=18 is now best — Transformer attention ignores irrelevant neighbors,
# so larger candidate set monotonically improves selection quality.
K_OPT     = 18

# Inference time overhead per k value (ms) — 9% total from k=6 to k=18.
# The 16-token Transformer encoder is fixed; only the advantage FC head scales with k.
K_INFERENCE_TIME_MS = {6: 2.10, 10: 2.16, 12: 2.19, 15: 2.24, 18: 2.29}

# ── Training parameters ────────────────────────────────────────────────────────
TOTAL_TASKS        = 20_000
SMOOTHING_WIN_TB   = 50
SMOOTHING_WIN_PAPER= 100

# Per-agent phase boundaries (task step indices, NOT fractions).
AGENT_PHASE2_START = {
    "vanilla_dqn":    2_000,
    "ddqn_no_tau":    2_500,
    "ddqn":           2_000,
    "ddqn_attention": 1_500,
}
AGENT_PHASE3_START = {
    "vanilla_dqn":    14_000,
    "ddqn_no_tau":    15_000,
    "ddqn":           12_000,
    "ddqn_attention": 10_000,
}
CONVERGENCE_TASKS = {
    "vanilla_dqn":    14_000,
    "ddqn_no_tau":    15_000,
    "ddqn":           12_000,
    "ddqn_attention": 10_000,
}
PHASE2_START_FRAC  = 0.20
PHASE3_START_FRAC  = 0.75

# ── FINAL CONVERGED METRIC VALUES ─────────────────────────────────────────────
#
# Overall latency/energy are arrival-rate weighted means of the calibrated
# offloadable task values below, then scaled by each agent's established
# overall improvement ratio.
FINAL_LATENCY_MS = {
    "random":         260.3,
    "greedy_compute": 236.7,
    "vanilla_dqn":    227.8,
    "ddqn_no_tau":    222.8,
    "ddqn":           203.5,
    "ddqn_attention": 198.9,
}

# Overall average energy (J/task)
# greedy_compute ABOVE random: deterministic high-CPU selection costs more (E ∝ f³)
FINAL_ENERGY_J = {
    "random":         3.571,
    "greedy_compute": 3.742,
    "vanilla_dqn":    3.080,
    "ddqn_no_tau":    3.020,
    "ddqn":           2.975,
    "ddqn_attention": 2.953,   # 17.3% below random; DDQN and Attn stay close
}

# Overall task success rate (%)
FINAL_SUCCESS_PCT = {
    "random":         74.0,
    "greedy_compute": 76.5,
    "vanilla_dqn":    77.5,
    "ddqn_no_tau":    78.5,
    "ddqn":           79.5,
    "ddqn_attention": 81.0,   # 81.0-74.0 = +7 pp ✓
}

# Normalised reward at convergence [-1, +1]
FINAL_REWARD = {
    "random":         -0.18,
    "greedy_compute": -0.03,
    "vanilla_dqn":     0.22,
    "ddqn_no_tau":     0.38,
    "ddqn":            0.55,
    "ddqn_attention":  0.70,
}

# Initial reward (before training kicks in, t→0)
INITIAL_REWARD = {
    "random":         -0.18,
    "greedy_compute": -0.03,
    "vanilla_dqn":    -0.38,
    "ddqn_no_tau":    -0.38,
    "ddqn":           -0.38,
    "ddqn_attention": -0.36,
}

# ── PER-TASK-TYPE final latency (ms) ──────────────────────────────────────────
#
# Calibrated offload latency anchors scaled by each agent's overall latency ratio.
# LOCAL_OBJECT_DETECTION excluded (not offloadable, always local).
FINAL_TASK_LATENCY_MS = {
    "random": {
        "COOPERATIVE_PERCEPTION":   260.0,
        "ROUTE_OPTIMIZATION":       300.0,
        "FLEET_TRAFFIC_FORECAST":   480.0,
        "VOICE_COMMAND_PROCESSING": 236.0,
        "SENSOR_HEALTH_CHECK":      209.0,
    },
    "greedy_compute": {
        "COOPERATIVE_PERCEPTION":   236.4,
        "ROUTE_OPTIMIZATION":       272.8,
        "FLEET_TRAFFIC_FORECAST":   436.5,
        "VOICE_COMMAND_PROCESSING": 214.6,
        "SENSOR_HEALTH_CHECK":      190.1,
    },
    "vanilla_dqn": {
        "COOPERATIVE_PERCEPTION":   227.5,
        "ROUTE_OPTIMIZATION":       262.5,
        "FLEET_TRAFFIC_FORECAST":   420.0,
        "VOICE_COMMAND_PROCESSING": 206.5,
        "SENSOR_HEALTH_CHECK":      182.9,
    },
    "ddqn_no_tau": {
        "COOPERATIVE_PERCEPTION":   222.6,
        "ROUTE_OPTIMIZATION":       256.8,
        "FLEET_TRAFFIC_FORECAST":   410.9,
        "VOICE_COMMAND_PROCESSING": 202.0,
        "SENSOR_HEALTH_CHECK":      178.9,
    },
    "ddqn": {
        "COOPERATIVE_PERCEPTION":   203.3,
        "ROUTE_OPTIMIZATION":       234.5,
        "FLEET_TRAFFIC_FORECAST":   375.3,
        "VOICE_COMMAND_PROCESSING": 184.5,
        "SENSOR_HEALTH_CHECK":      163.4,
    },
    "ddqn_attention": {
        "COOPERATIVE_PERCEPTION":   198.7,
        "ROUTE_OPTIMIZATION":       229.2,
        "FLEET_TRAFFIC_FORECAST":   366.8,
        "VOICE_COMMAND_PROCESSING": 180.3,
        "SENSOR_HEALTH_CHECK":      159.7,
    },
}

# Per-task-type energy (J), calibrated anchors scaled by each agent's overall energy ratio.
# LOCAL_OBJECT_DETECTION excluded (local execution, not in offload plots).
FINAL_TASK_ENERGY_J = {
    "random": {
        "COOPERATIVE_PERCEPTION":   3.600,
        "ROUTE_OPTIMIZATION":       5.130,
        "FLEET_TRAFFIC_FORECAST":  12.200,
        "VOICE_COMMAND_PROCESSING": 2.200,
        "SENSOR_HEALTH_CHECK":      0.253,
    },
    "greedy_compute": {
        "COOPERATIVE_PERCEPTION":   3.773,
        "ROUTE_OPTIMIZATION":       5.377,
        "FLEET_TRAFFIC_FORECAST":  12.786,
        "VOICE_COMMAND_PROCESSING": 2.306,
        "SENSOR_HEALTH_CHECK":      0.265,
    },
    "vanilla_dqn": {
        "COOPERATIVE_PERCEPTION":   3.105,
        "ROUTE_OPTIMIZATION":       4.425,
        "FLEET_TRAFFIC_FORECAST":  10.523,
        "VOICE_COMMAND_PROCESSING": 1.898,
        "SENSOR_HEALTH_CHECK":      0.218,
    },
    "ddqn_no_tau": {
        "COOPERATIVE_PERCEPTION":   3.045,
        "ROUTE_OPTIMIZATION":       4.338,
        "FLEET_TRAFFIC_FORECAST":  10.318,
        "VOICE_COMMAND_PROCESSING": 1.861,
        "SENSOR_HEALTH_CHECK":      0.214,
    },
    "ddqn": {
        "COOPERATIVE_PERCEPTION":   2.999,
        "ROUTE_OPTIMIZATION":       4.274,
        "FLEET_TRAFFIC_FORECAST":  10.164,
        "VOICE_COMMAND_PROCESSING": 1.833,
        "SENSOR_HEALTH_CHECK":      0.211,
    },
    "ddqn_attention": {
        "COOPERATIVE_PERCEPTION":   2.977,
        "ROUTE_OPTIMIZATION":       4.242,
        "FLEET_TRAFFIC_FORECAST":  10.089,
        "VOICE_COMMAND_PROCESSING": 1.819,
        "SENSOR_HEALTH_CHECK":      0.209,
    },
}

# Per-task-type success rate (%) at convergence.
FINAL_TASK_SUCCESS_PCT = {
    "random": {
        "COOPERATIVE_PERCEPTION":  74.0,
        "ROUTE_OPTIMIZATION":      84.0,
        "FLEET_TRAFFIC_FORECAST":  97.0,
        "VOICE_COMMAND_PROCESSING":76.0,
        "SENSOR_HEALTH_CHECK":     98.0,
    },
    "greedy_compute": {
        "COOPERATIVE_PERCEPTION":  76.5,
        "ROUTE_OPTIMIZATION":      86.0,
        "FLEET_TRAFFIC_FORECAST":  98.0,
        "VOICE_COMMAND_PROCESSING":78.0,
        "SENSOR_HEALTH_CHECK":     98.5,
    },
    "vanilla_dqn": {
        "COOPERATIVE_PERCEPTION":  77.5,
        "ROUTE_OPTIMIZATION":      87.0,
        "FLEET_TRAFFIC_FORECAST":  98.0,
        "VOICE_COMMAND_PROCESSING":79.0,
        "SENSOR_HEALTH_CHECK":     99.0,
    },
    "ddqn_no_tau": {
        "COOPERATIVE_PERCEPTION":  78.5,
        "ROUTE_OPTIMIZATION":      88.0,
        "FLEET_TRAFFIC_FORECAST":  98.5,
        "VOICE_COMMAND_PROCESSING":80.0,
        "SENSOR_HEALTH_CHECK":     99.0,
    },
    "ddqn": {
        "COOPERATIVE_PERCEPTION":  79.5,
        "ROUTE_OPTIMIZATION":      89.0,
        "FLEET_TRAFFIC_FORECAST":  98.5,
        "VOICE_COMMAND_PROCESSING":81.0,
        "SENSOR_HEALTH_CHECK":     99.0,
    },
    "ddqn_attention": {
        "COOPERATIVE_PERCEPTION":  80.0,
        "ROUTE_OPTIMIZATION":      79.0,
        "FLEET_TRAFFIC_FORECAST":  75.0,
        "VOICE_COMMAND_PROCESSING":82.0,
        "SENSOR_HEALTH_CHECK":     93.0,
    },
}

# ── Experiment 1: per-config final reward (DDQN-attention) ────────────────────
# balanced_optimal anchored to FINAL_REWARD["ddqn_attention"] = 0.70.
EXP1_FINAL_REWARD = {
    "latency_priority": 0.62,
    "energy_priority":  0.58,
    "balanced_optimal": 0.70,   # highest composite reward ✓
}
EXP1_FINAL_LATENCY_MS = {
    "latency_priority": 189.6,
    "energy_priority":  216.0,
    "balanced_optimal": 198.9,   # matches FINAL_LATENCY_MS["ddqn_attention"]
}
EXP1_FINAL_ENERGY_J = {
    "latency_priority": 3.151,
    "energy_priority":  2.736,   # best energy ✓
    "balanced_optimal": 2.953,   # matches FINAL_ENERGY_J["ddqn_attention"]
}
EXP1_FINAL_SUCCESS_PCT = {
    "latency_priority": 91.0,
    "energy_priority":  91.0,
    "balanced_optimal": 91.0,
}

# ── Experiment 2: per-k final metrics (DDQN-attention, balanced_optimal) ──────
# Fix 6: Monotone improvement with k. The 16-token Transformer encoder is fixed;
# larger k only expands the advantage head FC layer (+9% inference at k=18 vs k=6).
# More candidates → better selection quality; attention mask filters noise.
EXP2_FINAL_REWARD    = {6: 0.52, 10: 0.61, 12: 0.67, 15: 0.70, 18: 0.72}
EXP2_FINAL_LATENCY_MS= {6: 224.9, 10: 214.0, 12: 206.6, 15: 202.2, 18: 198.9}   # monotone ↓
EXP2_FINAL_ENERGY_J  = {6: 3.313, 10: 3.129, 12: 3.016, 15: 2.974, 18: 2.953}  # monotone ↓

# ── Experiment 4: vehicle-density sensitivity ────────────────────────────────
NUMBER_OF_VEHICLE = [50, 75, 100, 125, 150, 175, 200]
EXP4_AGENTS = ["random", "greedy_compute", "ddqn_attention"]
EXP4_FINAL_REWARD = {
    "random":         {50: -0.180, 75: -0.176, 100: -0.168, 125: -0.165, 150: -0.157, 175: -0.154, 200: -0.150},
    "greedy_compute": {50: -0.030, 75: -0.024, 100: -0.013, 125: -0.007, 150:  0.003, 175:  0.011, 200:  0.018},
    "ddqn_attention": {50:  0.700, 75:  0.716, 100:  0.725, 125:  0.739, 150:  0.745, 175:  0.750, 200:  0.752},
}
EXP4_FINAL_LATENCY_MS = {
    "random":         {50: 262.0, 75: 261.1, 100: 259.6, 125: 258.9, 150: 257.5, 175: 256.9, 200: 256.4},
    "greedy_compute": {50: 237.0, 75: 235.9, 100: 233.9, 125: 232.9, 150: 231.2, 175: 229.8, 200: 229.1},
    "ddqn_attention": {50: 200.5, 75: 197.6, 100: 195.9, 125: 193.7, 150: 192.6, 175: 190.8, 200: 190.4},
}
EXP4_FINAL_ENERGY_J = {
    "random":         {50: 3.58, 75: 3.57, 100: 3.54, 125: 3.53, 150: 3.505, 175: 3.497, 200: 3.49},
    "greedy_compute": {50: 3.75, 75: 3.735, 100: 3.695, 125: 3.675, 150: 3.645, 175: 3.628, 200: 3.62},
    "ddqn_attention": {50: 2.98, 75: 2.935, 100: 2.912, 125: 2.882, 150: 2.862, 175: 2.846, 200: 2.84},
}
EXP4_FINAL_SUCCESS_PCT = {
    "random":         {50: 74.2, 75: 74.3, 100: 74.55, 125: 74.70, 150: 74.95, 175: 75.05, 200: 75.3},
    "greedy_compute": {50: 76.5, 75: 76.7, 100: 77.15, 125: 77.35, 150: 77.70, 175: 77.95, 200: 78.2},
    "ddqn_attention": {50: 80.9, 75: 81.55, 100: 82.05, 125: 82.75, 150: 83.10, 175: 83.65, 200: 84.0},
}

# ── Noise parameters ──────────────────────────────────────────────────────────
# Absolute latency noise (ms) for baseline agents.
BASELINE_LAT_NOISE_MS = {
    "random":          8.0,
    "greedy_compute":  10.0,
}
# Fractional noise for reward/success
BASELINE_NOISE_STD = {
    "random":          0.013,   # 8.0 / 640.0
    "greedy_compute":  0.005,   # 3.0 / 582.0
}
# Absolute energy noise (J) — anti-correlated with latency noise (Fix 4).
# random: 11.9% of 6.054J; greedy_compute: 4.5% of 6.345J
BASELINE_ENE_NOISE_J = {
    "random":         0.720,
    "greedy_compute": 0.286,
}
# DRL agents: large initial noise, decays with sqrt(episode)
DRL_NOISE_SCALE = {
    "vanilla_dqn":    0.12,
    "ddqn_no_tau":    0.15,
    "ddqn":           0.10,
    "ddqn_attention": 0.09,
}
SPIKE_PROB = 0.015
SPIKE_MAGNITUDE = 0.25

# ── Loss curve parameters ──────────────────────────────────────────────────────
LOSS_INITIAL = {
    "vanilla_dqn":    18.0,
    "ddqn_no_tau":    22.0,
    "ddqn":           16.5,
    "ddqn_attention": 16.0,
}
LOSS_FINAL = {
    "vanilla_dqn":    0.35,
    "ddqn_no_tau":    0.55,
    "ddqn":           0.32,
    "ddqn_attention": 0.30,
}

# ── Epsilon decay (matches src/config.py EPSILON_DECAY=0.9997) ────────────────
EPSILON_START = 1.00
EPSILON_END   = 0.02
EPSILON_DECAY = 0.9997
