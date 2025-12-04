import torch
import os

class Config:
    # --- Paths ---
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "ddqn_iov_model.pth")
    PLOT_SAVE_PATH = os.path.join(BASE_DIR, "output", "training_results.png")
    LOG_DIR = os.path.join(BASE_DIR, "output", "runs")

    # --- Multi-RSU Settings (NEW) ---
    RSU_RANGE = 500          # Radius (Coverage is 1000m diameter)
    RSU_LOCATIONS = [        # (x, y) coordinates of RSUs
        (500, 0), 
        (1500, 0), 
        (2500, 0)
    ]
    MAP_WIDTH = 3000         # Total simulated world width
    MAP_HEIGHT = 1000        # Total simulated world height (-500 to +500)
    
    NUM_VEHICLES = 30        # Increased for larger map
    MAX_NEIGHBORS = 5        
    SEED = 42
    DT = 0.5                 # Physics time step (0.5s)

    # --- Task Physics ---
    TASK_SIZE_RANGE = (1, 5)      # MB
    CPU_CYCLES_RANGE = (100, 800) # Megacycles
    DEADLINE_RANGE = (0.5, 3.0)   # Seconds
    QOS_RANGE = (1, 3)            # 1=Standard, 2=High, 3=Critical

    # --- Network Physics ---
    BANDWIDTH_BASE = 20.0         # Mbps
    BANDWIDTH_VAR = 5.0           
    JITTER_STD = 0.05             

    # --- Vehicle Constraints ---
    MAX_SPEED = 30.0              # m/s (~108 km/h)
    MAX_ACCEL = 2.0               # m/s^2
    MAX_BATTERY = 100.0           # Percentage
    MAX_MEMORY = 4096             # MB
    BATTERY_DRAIN_RATE = 0.5      # % per task executed
    
    # --- Rewards ---
    REWARD_SUCCESS = 20
    REWARD_FAILURE = -20
    REWARD_HANDOVER_FAIL = -30    # Higher penalty for connection loss
    REWARD_SCALE = 10.0
    # Weights for multi-objective reward
    W_LATENCY = 0.6
    W_ENERGY = 0.2
    W_DEADLINE = 0.2

    # --- Dueling DDQN Settings ---
    # Vehicle Features (9): 
    # [CPU, Mem, Battery, Speed, PosX, PosY, Heading, Accel, Tasks]
    VEHICLE_FEAT_DIM = 9
    # State Dim: Task(4) + RSU(3) + Neighbors(9 * K)
    STATE_DIM = 4 + 3 + (VEHICLE_FEAT_DIM * MAX_NEIGHBORS)

    ACTION_DIM = MAX_NEIGHBORS + 2      # K vehicles + RSU + Drop/keep
    HIDDEN_DIM = 256
    LR = 0.0001
    GAMMA = 0.99
    TAU = 0.005 # Polyak Averaging factor
    EPSILON_START = 1.0
    EPSILON_END = 0.05
    EPSILON_DECAY = 0.995
    BATCH_SIZE = 64
    MEMORY_SIZE = 50000
    PER_ALPHA = 0.6  # How much prioritization to use (0=None, 1=Full)
    PER_BETA = 0.4   # Importance Sampling correction (annealed to 1.0)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"