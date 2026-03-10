import torch
import os
import json
import sys

class Config:
    # --- Paths ---
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Default values (will be overwritten by load_config)
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "best_ddqn_model.pth")
    PLOT_SAVE_PATH = os.path.join(BASE_DIR, "output", "training_results.png")
    LOG_DIR = os.path.join(BASE_DIR, "output", "runs")

    # --- IoV network Settings ---
    RSU_RANGE = 500
    RSU_LOCATIONS = [(500, 0), (1500, 0), (2500, 0)]
    MAP_WIDTH = 3000
    MAP_HEIGHT = 1000
    NUM_VEHICLES = 50
    MAX_NEIGHBORS = 5
    SEED = 42
    DT = 0.5

    # --- Task Physics ---
    TASK_SIZE_RANGE = (1, 6)
    CPU_CYCLES_RANGE = (100, 1000)
    DEADLINE_RANGE = (1.0, 5.0)
    QOS_RANGE = (1, 3)

    # --- Network Physics ---
    BANDWIDTH_BASE = 20.0
    BANDWIDTH_VAR = 5.0
    JITTER_STD = 0.05

    # --- Vehicle Constraints ---
    MAX_SPEED = 25.0
    MAX_ACCEL = 1.5
    MAX_BATTERY = 100.0
    MAX_MEMORY = 4096
    BATTERY_DRAIN_RATE = 0.2
    
    # --- Rewards ---
    REWARD_SUCCESS = 15
    REWARD_FAILURE = -15
    REWARD_HANDOVER_FAIL = -20
    REWARD_SCALE = 10.0
    W_LATENCY = 0.6
    W_ENERGY = 0.2
    W_DEADLINE = 0.2

    # --- Dueling DDQN Settings ---
    # Dummy env: [CPU, Mem, Battery, Speed, PosX, PosY, Heading, Accel, Tasks]
    VEHICLE_FEAT_DIM = 9
    # Dummy RSU features: [CPU, Mem, Bandwidth, Queue]
    RSU_FEAT_DIM = 4
    # Dummy: Task(4) + RSU(4) + Neighbors(9 * K)
    TASK_FEAT_DIM = 4
    STATE_DIM = TASK_FEAT_DIM + RSU_FEAT_DIM + (VEHICLE_FEAT_DIM * MAX_NEIGHBORS)

    # Dummy action space: K service vehicles + 1 RSU + 1 local
    ACTION_DIM = MAX_NEIGHBORS + 2

    # --- Redis Settings (populated by load_config when using redis_config.json) ---
    NUM_RSUS = 3
    RSU_IDS = ["RSU_0", "RSU_1", "RSU_2"]
    REDIS_HOST = "127.0.0.1"
    REDIS_PORT = 6379
    REDIS_POLL_INTERVAL = 0.05
    REDIS_RESULT_TIMEOUT = 30.0
    REDIS_TASK_FIELDS = ["mem_footprint_mb", "cpu_req_mcycles", "deadline_s", "qos"]
    REDIS_RSU_FIELDS = ["cpu_available", "memory_available", "queue_length", "cpu_utilization"]
    REDIS_VEHICLE_FIELDS = ["cpu_available", "mem_available", "cpu_utilization",
                            "mem_utilization", "queue_length", "speed", "heading", "distance_to_origin"]
    REDIS_NORMALIZATION = {}

    HIDDEN_DIM = 256
    EPISODS = 10000
    LR = 0.0001
    GAMMA = 0.99
    TAU = 0.005 # Polyak Averaging factor
    EPSILON_START = 1.0
    EPSILON_END = 0.02
    EPSILON_DECAY = 0.9997
    BATCH_SIZE = 256
    MEMORY_SIZE = 100000
    PER_ALPHA = 0.6  # How much prioritization to use (0=None, 1=Full)
    PER_BETA = 0.4   # Importance Sampling correction (annealed to 1.0)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- Legacy Database Settings (unused, kept for compatibility) ---
    DB_CONFIG = {}

    @classmethod
    def load_config(cls, config_name="config.json"):
        # 1. Load Common Config
        common_path = os.path.join(cls.BASE_DIR, "configs", "common_config.json")
        print(f"Loading common config from: {common_path}")
        with open(common_path, 'r') as f:
            _common = json.load(f)
            
        # 2. Load Specific Config
        config_path = os.path.join(cls.BASE_DIR, "configs", config_name)
        print(f"Loading specific config from: {config_path}")
        with open(config_path, 'r') as f:
            _specific = json.load(f)
            
        # Merge: Specific overrides Common
        # Simple recursive merge or just top-level update
        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
            
        _cfg = deep_update(_common, _specific)

        cls.MODEL_SAVE_PATH = os.path.join(cls.BASE_DIR, _cfg["system"]["model_save_path"])
        cls.PLOT_SAVE_PATH = os.path.join(cls.BASE_DIR, _cfg["system"]["plot_save_path"])
        cls.LOG_DIR = os.path.join(cls.BASE_DIR, _cfg["system"]["log_dir"])

        cls.RSU_RANGE = _cfg["iov_network"]["rsu_range"]
        cls.RSU_LOCATIONS = _cfg["iov_network"]["rsu_locations"]
        # Optional params (might not be present in all configs)
        cls.MAP_WIDTH = _cfg["iov_network"].get("map_width", 3000)
        cls.MAP_HEIGHT = _cfg["iov_network"].get("map_height", 1000)
        cls.NUM_VEHICLES = _cfg["iov_network"].get("num_vehicles", 50)
        
        cls.MAX_NEIGHBORS = _cfg["iov_network"]["max_neighbors"]
        cls.SEED = _cfg["system"]["seed"]
        cls.DT = _cfg["iov_network"]["dt"]

        cls.TASK_SIZE_RANGE = tuple(_cfg["task_physics"]["task_size_range"])
        cls.CPU_CYCLES_RANGE = tuple(_cfg["task_physics"]["cpu_cycles_range"])
        cls.DEADLINE_RANGE = tuple(_cfg["task_physics"]["deadline_range"])
        cls.QOS_RANGE = tuple(_cfg["task_physics"]["qos_range"])

        cls.BANDWIDTH_BASE = _cfg["network_physics"]["bandwidth_base"]
        cls.BANDWIDTH_VAR = _cfg["network_physics"]["bandwidth_var"]
        cls.JITTER_STD = _cfg["network_physics"]["jitter_std"]

        cls.MAX_SPEED = _cfg["vehicle_constraints"]["max_speed"]
        cls.MAX_ACCEL = _cfg["vehicle_constraints"]["max_accel"]
        cls.MAX_BATTERY = _cfg["vehicle_constraints"]["max_battery"]
        cls.MAX_MEMORY = _cfg["vehicle_constraints"]["max_memory"]
        cls.BATTERY_DRAIN_RATE = _cfg["vehicle_constraints"]["battery_drain_rate"]
        
        cls.REWARD_SUCCESS = _cfg["rewards"]["reward_success"]
        cls.REWARD_FAILURE = _cfg["rewards"]["reward_failure"]
        cls.REWARD_HANDOVER_FAIL = _cfg["rewards"]["reward_handover_fail"]
        cls.REWARD_SCALE = _cfg["rewards"]["reward_scale"]
        cls.W_LATENCY = _cfg["rewards"]["w_latency"]
        cls.W_ENERGY = _cfg["rewards"]["w_energy"]
        cls.W_DEADLINE = _cfg["rewards"]["w_deadline"]

        # Dynamic State/Action Dimensions
        if "redis" in _cfg:
            _redis = _cfg["redis"]
            cls.REDIS_HOST = _redis.get("host", "127.0.0.1")
            cls.REDIS_PORT = _redis.get("port", 6379)
            cls.REDIS_POLL_INTERVAL = _redis.get("poll_interval", 0.05)
            cls.REDIS_RESULT_TIMEOUT = _redis.get("result_timeout", 30.0)
            cls.NUM_RSUS = _redis.get("num_rsus", 1)
            cls.RSU_IDS = _redis.get("rsu_ids", [f"RSU_{i}" for i in range(cls.NUM_RSUS)])
            
            cols = _redis["state_columns"]
            cls.REDIS_TASK_FIELDS    = cols["task"]
            cls.REDIS_RSU_FIELDS     = cols["rsu"]
            cls.REDIS_VEHICLE_FIELDS = cols["vehicle"]
            cls.REDIS_NORMALIZATION  = _redis.get("normalization", {})
            
            cls.TASK_FEAT_DIM    = len(cls.REDIS_TASK_FIELDS)
            cls.RSU_FEAT_DIM     = len(cls.REDIS_RSU_FIELDS)
            cls.VEHICLE_FEAT_DIM = len(cls.REDIS_VEHICLE_FIELDS)
            # State: task features + (RSU features × num_rsus) + (vehicle features × max_neighbors)
            cls.STATE_DIM  = (cls.TASK_FEAT_DIM
                              + cls.RSU_FEAT_DIM * cls.NUM_RSUS
                              + cls.VEHICLE_FEAT_DIM * cls.MAX_NEIGHBORS)
            # Action: N RSUs + K service vehicles (no local — vehicle decides that itself)
            cls.ACTION_DIM = cls.NUM_RSUS + cls.MAX_NEIGHBORS
        else:
            # Dummy env: fixed feature dims
            cls.VEHICLE_FEAT_DIM = 9
            cls.TASK_FEAT_DIM    = 4
            cls.RSU_FEAT_DIM     = 4   # [cpu, mem, bandwidth, queue]
            cls.STATE_DIM  = cls.TASK_FEAT_DIM + cls.RSU_FEAT_DIM + (cls.VEHICLE_FEAT_DIM * cls.MAX_NEIGHBORS)
            cls.ACTION_DIM = cls.MAX_NEIGHBORS + 2  # K service vehicles + 1 RSU + 1 local
        cls.HIDDEN_DIM = _cfg["ddqn"]["hidden_dim"]
        cls.EPISODS = _cfg["ddqn"]["episodes"]
        cls.LR = _cfg["ddqn"]["lr"]
        cls.GAMMA = _cfg["ddqn"]["gamma"]
        cls.TAU = _cfg["ddqn"]["tau"]
        cls.EPSILON_START = _cfg["ddqn"]["epsilon_start"]
        cls.EPSILON_END = _cfg["ddqn"]["epsilon_end"]
        cls.EPSILON_DECAY = _cfg["ddqn"]["epsilon_decay"]
        cls.BATCH_SIZE = _cfg["ddqn"]["batch_size"]
        cls.MEMORY_SIZE = _cfg["ddqn"]["memory_size"]
        cls.PER_ALPHA = _cfg["ddqn"]["per_alpha"]
        cls.PER_BETA = _cfg["ddqn"]["per_beta"]
        
        cls.DEVICE = "cuda" if torch.cuda.is_available() and _cfg["system"]["device"] != "cpu" else "cpu"


# Auto-load default if not explicitly called (for backward compatibility or initial import)
# But we want main.py to control it.
# We'll check sys.argv here as a fallback or just load default.
if "--env" not in sys.argv:
    Config.load_config("config.json")

