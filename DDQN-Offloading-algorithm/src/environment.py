import numpy as np
import math
import random
from src.config import Config
from src.entities import Vehicle, Task, RSU

class IoVDummyEnv:
    def __init__(self):
        random.seed(Config.SEED)
        np.random.seed(Config.SEED)
        
        # 1. Initialize Multiple RSUs
        self.rsus = []
        for idx, (x, y) in enumerate(Config.RSU_LOCATIONS):
            self.rsus.append(RSU(idx, x, y, cpu_total=10000, memory_total=8192))
            
        # 2. Initialize Vehicles across the WHOLE map
        self.vehicles = [
            Vehicle(i, random.randint(2000, 5000), random.randint(1024, 4096)) 
            for i in range(Config.NUM_VEHICLES)
        ]
        
        # 3. Spread vehicles
        for v in self.vehicles:
            v.pos_x = random.uniform(0, Config.MAP_WIDTH)
            v.pos_y = random.uniform(-Config.RSU_RANGE, Config.RSU_RANGE) # Highway width
            v.speed = random.uniform(5, Config.MAX_SPEED) # Moving traffic
            v.heading = random.choice([0, 180]) + random.uniform(-10, 10) # East or West flow
            v.acceleration = 0

        self.active_rsu = None # The RSU handling the current step
        self.current_task = None
        self.task_origin_vehicle = None
        self.candidates = []

    def _get_closest_rsu(self, vehicle):
        """Finds the RSU with minimum distance to the vehicle."""
        closest = None
        min_dist = float('inf')
        
        for rsu in self.rsus:
            dist = math.sqrt((vehicle.pos_x - rsu.pos_x)**2 + (vehicle.pos_y - rsu.pos_y)**2)
            if dist < min_dist:
                min_dist = dist
                closest = rsu
                
        # Only connect if within range
        if min_dist <= Config.RSU_RANGE:
            return closest
        return None

    def reset(self):
        # 1. Randomly pick a vehicle that is CURRENTLY connected to an RSU
        valid_vehicles = []
        for v in self.vehicles:
            rsu = self._get_closest_rsu(v)
            if rsu:
                v.connected_rsu_id = rsu.rsu_id
                valid_vehicles.append((v, rsu))
        
        if not valid_vehicles:
            # Edge case: No vehicles in range (simulate ticks until one enters)
            self._update_physics_global()
            return self.reset()

        self.task_origin_vehicle, self.active_rsu = random.choice(valid_vehicles)
        
        # 2. Generate Task
        self.current_task = Task(
            task_id=random.randint(1000,9999),
            size=random.uniform(*Config.TASK_SIZE_RANGE),
            cpu_req=random.uniform(*Config.CPU_CYCLES_RANGE),
            vehicle_id=self.task_origin_vehicle.v_id,
            deadline=random.uniform(*Config.DEADLINE_RANGE),
            qos=random.randint(*Config.QOS_RANGE), 
            created_time=0
        )
        
        # 3. Global Physics Step (Background movement)
        self._update_physics_global()
        
        return self._get_state()

    def _update_physics_global(self):
        """Moves ALL vehicles in the map."""
        for v in self.vehicles:
            # Kinematics
            v.speed += v.acceleration * Config.DT
            v.speed = max(0, min(v.speed, Config.MAX_SPEED))
            
            rad = math.radians(v.heading)
            v.pos_x += v.speed * math.cos(rad) * Config.DT
            v.pos_y += v.speed * math.sin(rad) * Config.DT
            
            # Map Wrap-around (Infinite Highway Loop)
            if v.pos_x > Config.MAP_WIDTH: v.pos_x = 0
            if v.pos_x < 0: v.pos_x = Config.MAP_WIDTH
            if v.pos_y > 500: v.pos_y = 500
            if v.pos_y < -500: v.pos_y = -500

    def _get_state(self):
        # 1. Identify Candidates inside the ACTIVE RSU's range
        # Only vehicles connected to THIS RSU are valid candidates
        potential_candidates = []
        for v in self.vehicles:
            dist = math.sqrt((v.pos_x - self.active_rsu.pos_x)**2 + (v.pos_y - self.active_rsu.pos_y)**2)
            if dist <= Config.RSU_RANGE:
                potential_candidates.append(v)
        
        # Sort by signal strength (distance to Active RSU)
        self.candidates = sorted(
            potential_candidates, 
            key=lambda v: (v.pos_x - self.active_rsu.pos_x)**2 + (v.pos_y - self.active_rsu.pos_y)**2
        )[:Config.MAX_NEIGHBORS]
        
        # 2. Build RELATIVE State Vector
        state = []
        state.extend([self.current_task.size, self.current_task.cpu_req, self.current_task.deadline, self.current_task.qos])
        state.extend(self.active_rsu.to_feature_vector())
        
        for v in self.candidates:
            # IMPORTANT: Use Relative Features!
            state.extend(v.to_relative_feature_vector(self.active_rsu.pos_x, self.active_rsu.pos_y))
            
        # Padding
        expected_len = Config.STATE_DIM
        current_len = len(state)
        if current_len < expected_len:
            state.extend([0] * (expected_len - current_len))
            
        return np.array(state, dtype=np.float32)

    def get_action_mask(self):
        """
        Returns a binary mask [1, 0, 1, ...] of size ACTION_DIM.
        1 = Valid Action, 0 = Invalid Action.
        """
        mask = np.zeros(Config.ACTION_DIM, dtype=np.float32)
        
        # 1. Vehicle Candidates (Neighbors)
        for i, v in enumerate(self.candidates):
            if v.battery_avail > 5.0 and v.memory_avail > self.current_task.size:
                mask[i] = 1.0
        
        # 2. RSU (Infrastructure)
        mask[Config.MAX_NEIGHBORS] = 1.0
        
        # 3. Local Execution
        if self.task_origin_vehicle.battery_avail > 2.0:
            mask[Config.MAX_NEIGHBORS + 1] = 1.0
        
        return mask

    def step(self, action):
        done = True
        
        # --- HANDOVER CHECK (The "Risk" Logic) ---
        # Did the Task Vehicle leave the RSU coverage during this step?
        dist_now = math.sqrt((self.task_origin_vehicle.pos_x - self.active_rsu.pos_x)**2 + 
                             (self.task_origin_vehicle.pos_y - self.active_rsu.pos_y)**2)
        
        if dist_now > Config.RSU_RANGE:
            # FAILURE: Vehicle left the RSU before task completion
            # The agent receives a heavy penalty. It learns to avoid vehicles near the edge.
            return self._get_state(), Config.REWARD_HANDOVER_FAIL * self.current_task.qos, True, \
                   {"latency": 10.0, "success": 0, "reason": "Handover_Fail"}

        # --- Standard Logic ---
        bandwidth = max(1.0, np.random.normal(Config.BANDWIDTH_BASE, Config.BANDWIDTH_VAR))
        jitter = abs(np.random.normal(0, Config.JITTER_STD))
        
        latency = 0.0
        energy = 0.0
        
        # --- CASE 1: Offload to Neighbor ---
        if action < len(self.candidates):
            target = self.candidates[action]
            
            # Constraints Check (Redundant if masked, but good for safety)
            if target.battery_avail < 5.0 or target.memory_avail < self.current_task.size:
                 return self._get_state(), Config.REWARD_FAILURE * self.current_task.qos, True, \
                        {"latency": 10.0, "success": 0, "reason": "Constraint_Fail"}

            s_dist = math.sqrt((target.pos_x - self.active_rsu.pos_x)**2 + (target.pos_y - self.active_rsu.pos_y)**2)
            if s_dist > Config.RSU_RANGE:
                 return self._get_state(), Config.REWARD_HANDOVER_FAIL * self.current_task.qos, True, \
                        {"latency": 10.0, "success": 0, "reason": "Service_Left_Range"}

            # Stability Logic
            heading_diff = abs(target.heading - self.task_origin_vehicle.heading)
            if heading_diff > 180: heading_diff = 360 - heading_diff
            stability_factor = 1.0 - (heading_diff / 180.0)
            
            effective_bw = bandwidth * (0.5 + 0.5 * stability_factor)
            
            transmission_time = (self.current_task.size * 8) / effective_bw
            processing_time = self.current_task.cpu_req / max(1, target.cpu_avail)
            
            latency = transmission_time + processing_time + jitter
            energy = self.current_task.cpu_req * 0.002 # Remote energy cost (Transmission)
            target.battery_avail -= Config.BATTERY_DRAIN_RATE

        # --- CASE 2: Offload to RSU ---
        elif action == Config.MAX_NEIGHBORS:
            # RSU
            processing_time = self.current_task.cpu_req / max(1, self.active_rsu.cpu_avail)
            transmission_time = (self.current_task.size * 8) / (bandwidth * 1.5)
            latency = transmission_time + processing_time + jitter
            energy = self.current_task.cpu_req * 0.001 # RSU transmission is cheaper
        
        # --- CASE 3: Local Execution ---
        elif action == Config.MAX_NEIGHBORS + 1:
            # No Transmission time, ONLY Processing time
            # But Local CPU is usually weaker than RSU
            processing_time = self.current_task.cpu_req / max(1, self.task_origin_vehicle.cpu_avail)
            
            latency = processing_time # + 0 transmission
            
            # Local Energy is HIGH because we use our own CPU
            energy = self.current_task.cpu_req * 0.005 
            self.task_origin_vehicle.battery_avail -= (Config.BATTERY_DRAIN_RATE * 2) # Drains battery faster
        else:
            # Invalid Action Fallback
            return self._get_state(), Config.REWARD_FAILURE, True, {"latency": 10.0, "success": 0}

        deadline_met = latency <= self.current_task.deadline
        if deadline_met:
            rew_lat = Config.W_LATENCY * (1.0 - min(latency/2.0, 1.0))
            rew_ene = Config.W_ENERGY * (1.0 - min(energy/5.0, 1.0))
            rew_dead = Config.W_DEADLINE * 1.0
            reward = (rew_lat + rew_ene + rew_dead) * 10 * self.current_task.qos
        else:
            reward = Config.REWARD_FAILURE * self.current_task.qos

        # Normalize reward to keep gradients stable
        reward = reward / Config.REWARD_SCALE
        info = {
            "latency": latency,
            "energy": energy,
            "success": 1 if deadline_met else 0
        }
        
        return self._get_state(), reward, done, info