import torch
import os
import json
import sys

class Config:
    # --- Paths ---
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Default values (overwritten by load_config)
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "best_ddqn_model.pth")
    PLOT_SAVE_PATH  = os.path.join(BASE_DIR, "output", "training_results.png")
    LOG_DIR         = os.path.join(BASE_DIR, "output", "runs")

    # --- IoV network Settings (dummy env) ---
    RSU_RANGE     = 500
    RSU_LOCATIONS = [(500, 0), (1500, 0), (2500, 0)]
    MAP_WIDTH     = 3000
    MAP_HEIGHT    = 1000
    NUM_VEHICLES  = 50
    MAX_NEIGHBORS = 12  # Increased from 5 to handle multi-RSU candidates
    SEED          = 42
    DT            = 0.5

    # --- Task Physics (dummy env) ---
    TASK_SIZE_RANGE  = (1, 6)
    CPU_CYCLES_RANGE = (100, 1000)
    DEADLINE_RANGE   = (1.0, 5.0)
    QOS_RANGE        = (1, 3)

    # --- Network Physics (dummy env) ---
    BANDWIDTH_BASE = 20.0
    BANDWIDTH_VAR  = 5.0
    JITTER_STD     = 0.05

    # --- Vehicle Constraints (dummy env) ---
    MAX_SPEED         = 25.0
    MAX_ACCEL         = 1.5
    MAX_BATTERY       = 100.0
    MAX_MEMORY        = 4096
    BATTERY_DRAIN_RATE = 0.2

    # --- Rewards ---
    REWARD_SUCCESS      = 15
    REWARD_FAILURE      = -15
    REWARD_HANDOVER_FAIL = -20
    REWARD_SCALE        = 10.0
    W_LATENCY           = 0.6
    W_ENERGY            = 0.2
    W_DEADLINE          = 0.2

    # --- Dueling DDQN Settings ---
    VEHICLE_FEAT_DIM = 9
    RSU_FEAT_DIM     = 4
    TASK_FEAT_DIM    = 4
    STATE_DIM        = TASK_FEAT_DIM + RSU_FEAT_DIM + (VEHICLE_FEAT_DIM * MAX_NEIGHBORS)
    ACTION_DIM       = MAX_NEIGHBORS + 2  # dummy: K svs + 1 RSU + 1 local

    # --- Redis Settings (populated by load_config when using redis_config.json) ---
    NUM_RSUS       = 3
    RSU_IDS        = ["RSU_0", "RSU_1", "RSU_2"]
    DRL_INSTANCES  = [{"instance_id": 0, "rsu_id": "RSU_0", "redis_db": 0, "active": True}]
    REDIS_HOST           = "127.0.0.1"
    REDIS_PORT           = 6379
    REDIS_POLL_INTERVAL  = 0.05
    REDIS_RESULT_TIMEOUT = 30.0
    REDIS_TASK_FIELDS    = ["mem_footprint_mb", "cpu_req_mcycles", "deadline_s", "qos"]
    REDIS_RSU_FIELDS     = ["cpu_available", "memory_available", "queue_length", "cpu_utilization"]
    REDIS_VEHICLE_FIELDS = ["cpu_available", "mem_available", "cpu_utilization",
                            "mem_utilization", "queue_length", "speed", "heading",
                            "acceleration", "distance_to_origin"]
    REDIS_NORMALIZATION  = {}

    HIDDEN_DIM    = 256
    EPISODS       = 10000
    LR            = 0.0001
    GAMMA         = 0.99
    TAU           = 0.005
    EPSILON_START = 1.0
    EPSILON_END   = 0.02
    EPSILON_DECAY = 0.9997
    BATCH_SIZE    = 256
    MEMORY_SIZE   = 100000
    PER_ALPHA     = 0.6
    PER_BETA      = 0.4
    DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Legacy (unused) ---
    DB_CONFIG = {}

    @classmethod
    def load_config(cls, config_name="dummy_config.json"):
        """Load a single self-contained config file. No common-config merging."""
        config_path = os.path.join(cls.BASE_DIR, "configs", config_name)
        print(f"Loading config from: {config_path}")
        with open(config_path, 'r') as f:
            _cfg = json.load(f)

        # ── System paths & runtime settings ──────────────────────────────────
        _sys = _cfg.get("system", {})
        if "model_save_path" in _sys:
            cls.MODEL_SAVE_PATH = os.path.join(cls.BASE_DIR, _sys["model_save_path"])
        if "plot_save_path" in _sys:
            cls.PLOT_SAVE_PATH  = os.path.join(cls.BASE_DIR, _sys["plot_save_path"])
        if "log_dir" in _sys:
            cls.LOG_DIR = os.path.join(cls.BASE_DIR, _sys["log_dir"])
        cls.SEED   = _sys.get("seed", 42)
        cls.DEVICE = "cuda" if torch.cuda.is_available() and _sys.get("device", "auto") != "cpu" else "cpu"

        # ── Dummy-env-only sections (guarded — not required for redis mode) ──
        if "iov_network" in _cfg:
            _iov = _cfg["iov_network"]
            cls.RSU_RANGE     = _iov.get("rsu_range", cls.RSU_RANGE)
            cls.RSU_LOCATIONS = _iov.get("rsu_locations", cls.RSU_LOCATIONS)
            cls.MAP_WIDTH     = _iov.get("map_width", cls.MAP_WIDTH)
            cls.MAP_HEIGHT    = _iov.get("map_height", cls.MAP_HEIGHT)
            cls.NUM_VEHICLES  = _iov.get("num_vehicles", cls.NUM_VEHICLES)
            cls.MAX_NEIGHBORS = _iov.get("max_neighbors", cls.MAX_NEIGHBORS)
            cls.DT            = _iov.get("dt", cls.DT)

        if "task_physics" in _cfg:
            _tp = _cfg["task_physics"]
            cls.TASK_SIZE_RANGE  = tuple(_tp["task_size_range"])
            cls.CPU_CYCLES_RANGE = tuple(_tp["cpu_cycles_range"])
            cls.DEADLINE_RANGE   = tuple(_tp["deadline_range"])
            cls.QOS_RANGE        = tuple(_tp["qos_range"])

        if "network_physics" in _cfg:
            _np = _cfg["network_physics"]
            cls.BANDWIDTH_BASE = _np["bandwidth_base"]
            cls.BANDWIDTH_VAR  = _np["bandwidth_var"]
            cls.JITTER_STD     = _np["jitter_std"]

        if "vehicle_constraints" in _cfg:
            _vc = _cfg["vehicle_constraints"]
            cls.MAX_SPEED          = _vc["max_speed"]
            cls.MAX_ACCEL          = _vc["max_accel"]
            cls.MAX_BATTERY        = _vc["max_battery"]
            cls.MAX_MEMORY         = _vc["max_memory"]
            cls.BATTERY_DRAIN_RATE = _vc["battery_drain_rate"]

        # ── Rewards (required in both modes) ─────────────────────────────────
        _rew = _cfg["rewards"]
        cls.REWARD_SUCCESS      = _rew["reward_success"]
        cls.REWARD_FAILURE      = _rew["reward_failure"]
        cls.REWARD_HANDOVER_FAIL = _rew["reward_handover_fail"]
        cls.REWARD_SCALE        = _rew["reward_scale"]
        cls.W_LATENCY           = _rew["w_latency"]
        cls.W_ENERGY            = _rew["w_energy"]
        cls.W_DEADLINE          = _rew["w_deadline"]

        # ── DDQN hyperparameters (required in both modes) ────────────────────
        _ddqn = _cfg["ddqn"]
        cls.HIDDEN_DIM    = _ddqn["hidden_dim"]
        cls.EPISODS       = _ddqn["episodes"]
        cls.LR            = _ddqn["lr"]
        cls.GAMMA         = _ddqn["gamma"]
        cls.TAU           = _ddqn["tau"]
        cls.EPSILON_START = _ddqn["epsilon_start"]
        cls.EPSILON_END   = _ddqn["epsilon_end"]
        cls.EPSILON_DECAY = _ddqn["epsilon_decay"]
        cls.BATCH_SIZE    = _ddqn["batch_size"]
        cls.MEMORY_SIZE   = _ddqn["memory_size"]
        cls.PER_ALPHA     = _ddqn["per_alpha"]
        cls.PER_BETA      = _ddqn["per_beta"]

        # ── Redis section (only for redis mode) ──────────────────────────────
        if "redis" in _cfg:
            _redis = _cfg["redis"]
            cls.REDIS_HOST           = _redis.get("host", "127.0.0.1")
            cls.REDIS_PORT           = _redis.get("port", 6379)
            cls.REDIS_POLL_INTERVAL  = _redis.get("poll_interval", 0.05)
            cls.REDIS_RESULT_TIMEOUT = _redis.get("result_timeout", 30.0)
            cls.NUM_RSUS             = _redis.get("num_rsus", 1)
            cls.RSU_IDS              = _redis.get("rsu_ids", [f"RSU_{i}" for i in range(cls.NUM_RSUS)])
            # max_neighbors in redis section overrides iov_network value
            cls.MAX_NEIGHBORS        = _redis.get("max_neighbors", cls.MAX_NEIGHBORS)

            cols = _redis["state_columns"]
            cls.REDIS_TASK_FIELDS    = cols["task"]
            cls.REDIS_RSU_FIELDS     = cols["rsu"]
            cls.REDIS_VEHICLE_FIELDS = cols["vehicle"]
            cls.REDIS_NORMALIZATION  = _redis.get("normalization", {})
            cls.DRL_INSTANCES        = _redis.get("agent_instances",
                                         _redis.get("drl_instances", [  # fallback for old configs
                {"instance_id": 0, "rsu_id": cls.RSU_IDS[0], "redis_db": 0, "active": True}
            ]))

            # Recompute state/action dims for redis mode
            cls.TASK_FEAT_DIM    = len(cls.REDIS_TASK_FIELDS)
            cls.RSU_FEAT_DIM     = len(cls.REDIS_RSU_FIELDS)
            cls.VEHICLE_FEAT_DIM = len(cls.REDIS_VEHICLE_FIELDS)
            # State: task + (RSU × num_rsus) + (vehicle × max_neighbors)
            cls.STATE_DIM  = (cls.TASK_FEAT_DIM
                              + cls.RSU_FEAT_DIM  * cls.NUM_RSUS
                              + cls.VEHICLE_FEAT_DIM * cls.MAX_NEIGHBORS)
            # Action: N RSUs + K service vehicles
            cls.ACTION_DIM = cls.NUM_RSUS + cls.MAX_NEIGHBORS
        else:
            # Dummy env dims
            cls.VEHICLE_FEAT_DIM = 9
            cls.TASK_FEAT_DIM    = 4
            cls.RSU_FEAT_DIM     = 4
            cls.STATE_DIM  = cls.TASK_FEAT_DIM + cls.RSU_FEAT_DIM + (cls.VEHICLE_FEAT_DIM * cls.MAX_NEIGHBORS)
            cls.ACTION_DIM = cls.MAX_NEIGHBORS + 2


# Auto-load dummy config at import time (main.py overrides this for redis mode)
if "--env" not in sys.argv:
    Config.load_config("dummy_config.json")
