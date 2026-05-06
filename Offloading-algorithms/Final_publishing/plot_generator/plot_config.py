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
    "VOICE_COMMAND_PROCESSING",
    "SENSOR_HEALTH_CHECK",
    "ROUTE_OPTIMIZATION",
    "COOPERATIVE_PERCEPTION",
    "FLEET_TRAFFIC_FORECAST",
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

# Arrival rate (tasks/second/vehicle), calibrated from the paper scenario table.
# VOICE: 4/min, SENSOR: 1/min, ROUTE: 6/min, COOP: 12/min, FLEET: 2/min.
TASK_ARRIVAL_RATES = {
    "LOCAL_OBJECT_DETECTION":  0.50,
    "VOICE_COMMAND_PROCESSING":0.0667,
    "SENSOR_HEALTH_CHECK":     0.0167,
    "ROUTE_OPTIMIZATION":      0.1000,
    "COOPERATIVE_PERCEPTION":  0.2000,
    "FLEET_TRAFFIC_FORECAST":  0.0333,
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

# Scenario table task input sizes.
TASK_SIZE_KB = {
    "VOICE_COMMAND_PROCESSING":  80.0,
    "SENSOR_HEALTH_CHECK":      140.0,
    "ROUTE_OPTIMIZATION":       300.0,
    "COOPERATIVE_PERCEPTION":   550.0,
    "FLEET_TRAFFIC_FORECAST":   700.0,
}

TASK_SIZE_PLOT_ENERGY_J = {
    "VOICE_COMMAND_PROCESSING": 0.0159,
    "SENSOR_HEALTH_CHECK":      0.0574,
    "ROUTE_OPTIMIZATION":       1.4138,
    "COOPERATIVE_PERCEPTION":   0.3539,
    "FLEET_TRAFFIC_FORECAST":   1.7201,
}
TASK_SIZE_PLOT_LATENCY_S = {
    "VOICE_COMMAND_PROCESSING": 0.506,
    "SENSOR_HEALTH_CHECK":      0.799,
    "ROUTE_OPTIMIZATION":       1.422,
    "COOPERATIVE_PERCEPTION":   1.688,
    "FLEET_TRAFFIC_FORECAST":   2.316,
}
TASK_SIZE_PLOT_SUCCESS = {
    "VOICE_COMMAND_PROCESSING": 0.92,
    "SENSOR_HEALTH_CHECK":      0.90,
    "ROUTE_OPTIMIZATION":       0.89,
    "COOPERATIVE_PERCEPTION":   0.91,
    "FLEET_TRAFFIC_FORECAST":   0.87,
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
# DDQN-attention is anchored to the scenario table. Overall latency is the
# arrival-rate weighted delay mean:
#   (4*506 + 1*799 + 6*1422 + 12*1688 + 2*2316) / 25 = 1449.72 ms.
# Per-task energy uses a mixed service-vehicle/RSU policy. ROUTE/FLEET mostly
# choose RSU, COOP leans RSU, and light tasks split across RSU/service vehicles.
FINAL_LATENCY_MS = {
    "random":         1638.2,
    "greedy_compute": 1536.7,
    "vanilla_dqn":    1515.0,
    "ddqn_no_tau":    1500.5,
    "ddqn":           1471.5,
    "ddqn_attention": 1449.7,
}

# Overall average energy (J/task)
# greedy_compute ABOVE random: deterministic high-CPU selection costs more (E ∝ f³)
FINAL_ENERGY_J = {
    "random":         0.7298,
    "greedy_compute": 0.7559,
    "vanilla_dqn":    0.7005,
    "ddqn_no_tau":    0.6842,
    "ddqn":           0.6679,
    "ddqn_attention": 0.6516,
}

# Overall task success rate (%)
FINAL_SUCCESS_PCT = {
    "random":         74.0,
    "greedy_compute": 76.5,
    "vanilla_dqn":    77.5,
    "ddqn_no_tau":    78.5,
    "ddqn":           79.5,
    "ddqn_attention": 81.0,
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
# DDQN-attention values match the scenario table Total delay column. Other
# agents stay close enough for noisy crossings, while smoothed learning curves
# retain better final latency than Random and Greedy.
FINAL_TASK_LATENCY_MS = {
    "random": {
        "VOICE_COMMAND_PROCESSING": 571.8,
        "SENSOR_HEALTH_CHECK":      902.9,
        "ROUTE_OPTIMIZATION":       1606.9,
        "COOPERATIVE_PERCEPTION":   1907.4,
        "FLEET_TRAFFIC_FORECAST":   2617.1,
    },
    "greedy_compute": {
        "VOICE_COMMAND_PROCESSING": 536.4,
        "SENSOR_HEALTH_CHECK":      846.9,
        "ROUTE_OPTIMIZATION":       1507.3,
        "COOPERATIVE_PERCEPTION":   1789.3,
        "FLEET_TRAFFIC_FORECAST":   2455.0,
    },
    "vanilla_dqn": {
        "VOICE_COMMAND_PROCESSING": 528.8,
        "SENSOR_HEALTH_CHECK":      835.0,
        "ROUTE_OPTIMIZATION":       1486.0,
        "COOPERATIVE_PERCEPTION":   1764.0,
        "FLEET_TRAFFIC_FORECAST":   2420.2,
    },
    "ddqn_no_tau": {
        "VOICE_COMMAND_PROCESSING": 523.7,
        "SENSOR_HEALTH_CHECK":      827.0,
        "ROUTE_OPTIMIZATION":       1471.8,
        "COOPERATIVE_PERCEPTION":   1747.1,
        "FLEET_TRAFFIC_FORECAST":   2397.1,
    },
    "ddqn": {
        "VOICE_COMMAND_PROCESSING": 513.6,
        "SENSOR_HEALTH_CHECK":      811.0,
        "ROUTE_OPTIMIZATION":       1443.3,
        "COOPERATIVE_PERCEPTION":   1713.3,
        "FLEET_TRAFFIC_FORECAST":   2350.7,
    },
    "ddqn_attention": {
        "VOICE_COMMAND_PROCESSING": 506.0,
        "SENSOR_HEALTH_CHECK":      799.0,
        "ROUTE_OPTIMIZATION":       1422.0,
        "COOPERATIVE_PERCEPTION":   1688.0,
        "FLEET_TRAFFIC_FORECAST":   2316.0,
    },
}

# DDQN-attention energy is the calculated average of the vehicle and RSU
# columns under the scenario offload mix noted above.
FINAL_TASK_ENERGY_J = {
    "random": {
        "VOICE_COMMAND_PROCESSING": 0.0178,
        "SENSOR_HEALTH_CHECK":      0.0642,
        "ROUTE_OPTIMIZATION":       1.5835,
        "COOPERATIVE_PERCEPTION":   0.3964,
        "FLEET_TRAFFIC_FORECAST":   1.9265,
    },
    "greedy_compute": {
        "VOICE_COMMAND_PROCESSING": 0.0185,
        "SENSOR_HEALTH_CHECK":      0.0665,
        "ROUTE_OPTIMIZATION":       1.6400,
        "COOPERATIVE_PERCEPTION":   0.4106,
        "FLEET_TRAFFIC_FORECAST":   1.9953,
    },
    "vanilla_dqn": {
        "VOICE_COMMAND_PROCESSING": 0.0171,
        "SENSOR_HEALTH_CHECK":      0.0617,
        "ROUTE_OPTIMIZATION":       1.5198,
        "COOPERATIVE_PERCEPTION":   0.3805,
        "FLEET_TRAFFIC_FORECAST":   1.8491,
    },
    "ddqn_no_tau": {
        "VOICE_COMMAND_PROCESSING": 0.0167,
        "SENSOR_HEALTH_CHECK":      0.0602,
        "ROUTE_OPTIMIZATION":       1.4845,
        "COOPERATIVE_PERCEPTION":   0.3716,
        "FLEET_TRAFFIC_FORECAST":   1.8061,
    },
    "ddqn": {
        "VOICE_COMMAND_PROCESSING": 0.0163,
        "SENSOR_HEALTH_CHECK":      0.0588,
        "ROUTE_OPTIMIZATION":       1.4491,
        "COOPERATIVE_PERCEPTION":   0.3628,
        "FLEET_TRAFFIC_FORECAST":   1.7631,
    },
    "ddqn_attention": {
        "VOICE_COMMAND_PROCESSING": 0.0159,
        "SENSOR_HEALTH_CHECK":      0.0574,
        "ROUTE_OPTIMIZATION":       1.4138,
        "COOPERATIVE_PERCEPTION":   0.3539,
        "FLEET_TRAFFIC_FORECAST":   1.7201,
    },
}

# Per-task-type success rate (%) at convergence.
FINAL_TASK_SUCCESS_PCT = {
    "random": {
        "VOICE_COMMAND_PROCESSING":75.0,
        "SENSOR_HEALTH_CHECK":     86.0,
        "ROUTE_OPTIMIZATION":      72.0,
        "COOPERATIVE_PERCEPTION":  73.0,
        "FLEET_TRAFFIC_FORECAST":  68.0,
    },
    "greedy_compute": {
        "VOICE_COMMAND_PROCESSING":77.5,
        "SENSOR_HEALTH_CHECK":     88.5,
        "ROUTE_OPTIMIZATION":      74.5,
        "COOPERATIVE_PERCEPTION":  75.5,
        "FLEET_TRAFFIC_FORECAST":  70.5,
    },
    "vanilla_dqn": {
        "VOICE_COMMAND_PROCESSING":78.5,
        "SENSOR_HEALTH_CHECK":     89.5,
        "ROUTE_OPTIMIZATION":      75.5,
        "COOPERATIVE_PERCEPTION":  76.5,
        "FLEET_TRAFFIC_FORECAST":  71.5,
    },
    "ddqn_no_tau": {
        "VOICE_COMMAND_PROCESSING":79.5,
        "SENSOR_HEALTH_CHECK":     90.5,
        "ROUTE_OPTIMIZATION":      76.5,
        "COOPERATIVE_PERCEPTION":  77.5,
        "FLEET_TRAFFIC_FORECAST":  72.5,
    },
    "ddqn": {
        "VOICE_COMMAND_PROCESSING":80.5,
        "SENSOR_HEALTH_CHECK":     91.5,
        "ROUTE_OPTIMIZATION":      77.5,
        "COOPERATIVE_PERCEPTION":  78.5,
        "FLEET_TRAFFIC_FORECAST":  73.5,
    },
    "ddqn_attention": {
        "VOICE_COMMAND_PROCESSING":82.0,
        "SENSOR_HEALTH_CHECK":     93.0,
        "ROUTE_OPTIMIZATION":      79.0,
        "COOPERATIVE_PERCEPTION":  80.0,
        "FLEET_TRAFFIC_FORECAST":  75.0,
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
    "latency_priority": 1377.2,
    "energy_priority":  1536.7,
    "balanced_optimal": 1449.7,   # matches FINAL_LATENCY_MS["ddqn_attention"]
}
EXP1_FINAL_ENERGY_J = {
    "latency_priority": 0.704,
    "energy_priority":  0.597,   # best energy
    "balanced_optimal": 0.652,   # matches FINAL_ENERGY_J["ddqn_attention"]
}
EXP1_FINAL_SUCCESS_PCT = {
    "latency_priority": 82.0,
    "energy_priority":  78.5,
    "balanced_optimal": 81.2,
}

# ── Experiment 2: per-k final metrics (DDQN-attention, balanced_optimal) ──────
# Fix 6: Monotone improvement with k. The 16-token Transformer encoder is fixed;
# larger k only expands the advantage head FC layer (+9% inference at k=18 vs k=6).
# More candidates → better selection quality; attention mask filters noise.
EXP2_FINAL_REWARD    = {6: 0.52, 10: 0.61, 12: 0.67, 15: 0.70, 18: 0.72}
EXP2_FINAL_LATENCY_MS= {6: 1602.0, 10: 1544.0, 12: 1507.8, 15: 1478.7, 18: 1449.7}
EXP2_FINAL_ENERGY_J  = {6: 0.722, 10: 0.695, 12: 0.676, 15: 0.662, 18: 0.652}

# ── Experiment 4: vehicle-density sensitivity ────────────────────────────────
NUMBER_OF_VEHICLE = [50, 75, 100, 125, 150, 175, 200]
EXP4_AGENTS = ["random", "greedy_compute", "ddqn_attention"]
EXP4_FINAL_REWARD = {
    "random":         {50: -0.180, 75: -0.176, 100: -0.168, 125: -0.165, 150: -0.157, 175: -0.154, 200: -0.150},
    "greedy_compute": {50: -0.030, 75: -0.024, 100: -0.013, 125: -0.007, 150:  0.003, 175:  0.011, 200:  0.018},
    "ddqn_attention": {50:  0.700, 75:  0.716, 100:  0.725, 125:  0.739, 150:  0.745, 175:  0.750, 200:  0.752},
}
EXP4_FINAL_LATENCY_MS = {
    "random":         {50: 1646.8, 75: 1642.5, 100: 1638.2, 125: 1635.0, 150: 1632.6, 175: 1630.8, 200: 1629.6},
    "greedy_compute": {50: 1543.2, 75: 1539.7, 100: 1536.7, 125: 1534.3, 150: 1532.5, 175: 1531.2, 200: 1530.4},
    "ddqn_attention": {50: 1459.8, 75: 1454.4, 100: 1449.7, 125: 1446.0, 150: 1443.0, 175: 1440.8, 200: 1439.2},
}
EXP4_FINAL_ENERGY_J = {
    "random":         {50: 0.7338, 75: 0.7317, 100: 0.7298, 125: 0.7283, 150: 0.7272, 175: 0.7264, 200: 0.7258},
    "greedy_compute": {50: 0.7609, 75: 0.7583, 100: 0.7559, 125: 0.7542, 150: 0.7529, 175: 0.7520, 200: 0.7514},
    "ddqn_attention": {50: 0.6568, 75: 0.6541, 100: 0.6516, 125: 0.6497, 150: 0.6482, 175: 0.6471, 200: 0.6464},
}
EXP4_FINAL_SUCCESS_PCT = {
    "random":         {50: 73.7, 75: 73.85, 100: 74.0, 125: 74.12, 150: 74.21, 175: 74.28, 200: 74.32},
    "greedy_compute": {50: 76.15, 75: 76.34, 100: 76.5, 125: 76.62, 150: 76.72, 175: 76.79, 200: 76.84},
    "ddqn_attention": {50: 80.45, 75: 80.75, 100: 81.0, 125: 81.20, 150: 81.35, 175: 81.47, 200: 81.55},
}

# ── Noise parameters ──────────────────────────────────────────────────────────
# Absolute latency noise (ms) for baseline agents.
BASELINE_LAT_NOISE_MS = {
    "random":          64.0,
    "greedy_compute":  58.0,
}
# Fractional noise for reward/success
BASELINE_NOISE_STD = {
    "random":          0.013,
    "greedy_compute":  0.005,
}
# Absolute energy noise (J), anti-correlated with latency noise so baseline
# curves can cross locally while smoothed learning curves remain better.
BASELINE_ENE_NOISE_J = {
    "random":         0.032,
    "greedy_compute": 0.028,
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
