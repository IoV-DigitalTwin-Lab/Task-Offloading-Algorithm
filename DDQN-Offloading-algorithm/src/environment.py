import numpy as np
import math
import random
import time
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from src.config import Config
from src.entities import Vehicle, Task, RSU

class IoVDummyEnv:
    def __init__(self):
        random.seed(Config.SEED)
        np.random.seed(Config.SEED)
        
        # 1. Initialize Multiple RSUs
        self.rsus = []
        for idx, (x, y) in enumerate(Config.RSU_LOCATIONS):
            rsu = RSU(idx, x, y, cpu_total=10000, memory_total=8192)
            # Initialize utilization metrics for dummy env
            rsu.cpu_utilization = 0.0
            rsu.memory_utilization = 0.0
            rsu.processing_count = 0
            rsu.max_concurrent_tasks = 10
            self.rsus.append(rsu)
            
        # 2. Initialize Vehicles across the WHOLE map
        self.vehicles = []
        for i in range(Config.NUM_VEHICLES):
            v = Vehicle(i, random.randint(2000, 5000), random.randint(1024, 4096))
            # Initialize utilization metrics for dummy env
            v.cpu_utilization = 0.0
            v.mem_utilization = 0.0
            v.processing_count = 0
            self.vehicles.append(v)
        
        self._respawn_vehicles() # Initial Spawn

        self.active_rsu = None # The RSU handling the current step
        self.current_task = None
        self.task_origin_vehicle = None
        self.candidates = []

    def _respawn_vehicles(self):
        """Helper to randomize vehicle positions across the map."""
        for v in self.vehicles:
            v.pos_x = random.uniform(0, Config.MAP_WIDTH)
            v.pos_y = random.uniform(-Config.RSU_RANGE, Config.RSU_RANGE)
            v.speed = random.uniform(5, Config.MAX_SPEED)
            v.heading = random.choice([0, 180]) + random.uniform(-10, 10)
            v.acceleration = 0

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
        # --- ROBUST RESET LOGIC ---
        # Instead of recursion, we loop until we find a valid scenario.
        attempts = 0
        
        while True:
            valid_vehicles = []
            for v in self.vehicles:
                rsu = self._get_closest_rsu(v)
                if rsu:
                    v.connected_rsu_id = rsu.rsu_id
                    valid_vehicles.append((v, rsu))
            
            # If we found vehicles, break the loop
            if valid_vehicles:
                break
            
            # If no vehicles are in range, advance physics and try again
            self._update_physics_global()
            attempts += 1
            
            # FAILSAFE: If simulation runs empty for 1000 steps, respawn everyone.
            # This prevents infinite loops or recursion errors.
            if attempts > 1000:
                self._respawn_vehicles()
                attempts = 0

        # Select a random valid vehicle and its RSU
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

class IoVDatabaseEnv:
    def __init__(self):
        self.conn = psycopg2.connect(
            host=Config.DB_CONFIG["host"],
            port=Config.DB_CONFIG["port"],
            user=Config.DB_CONFIG["user"],
            password=Config.DB_CONFIG["password"],
            dbname=Config.DB_CONFIG["dbname"]
        )
        self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        
        # Initialize state variables
        self.current_task = None
        self.active_rsu = None
        self.task_origin_vehicle = None
        self.candidates = []
        
        # Track processed requests to avoid re-processing
        self.last_processed_request_id = -1

    def _fetch_latest_request(self):
        """Polls the database for the next unprocessed offloading request."""
        while True:
            query = """
                SELECT * FROM offloading_requests 
                WHERE id > %s 
                ORDER BY id ASC 
                LIMIT 1
            """
            self.cursor.execute(query, (self.last_processed_request_id,))
            row = self.cursor.fetchone()
            
            if row:
                self.last_processed_request_id = row['id']
                return row
            
            # Wait before polling again
            time.sleep(Config.DB_CONFIG["poll_interval"])

    def _fetch_vehicle_status(self, vehicle_id, request_time):
        """
        Fetches the vehicle status closest to the request time (Time Travel).
        This ensures we train on the state AS IT WAS when the request happened.
        """
        # We look for the status update immediately preceding or equal to the request time
        query = """
            SELECT * FROM vehicle_status 
            WHERE vehicle_id = %s AND update_time <= %s
            ORDER BY update_time DESC 
            LIMIT 1
        """
        self.cursor.execute(query, (vehicle_id, request_time))
        row = self.cursor.fetchone()
        
        if row:
            print(f"[DB] Found status for {vehicle_id} at {row['update_time']} (Req: {request_time})")
            return row
        else:
            print(f"[DB-WARN] No status found for {vehicle_id} before {request_time}. Using fallback.")
            return None

    def _fetch_neighbors(self, rsu_id, origin_vehicle_id, origin_x, origin_y, request_time):
        """
        Fetches potential candidate vehicles connected to the same RSU.
        Prioritizes:
        1. Distance to Origin Vehicle (Ascending)
        2. CPU Availability (Descending)
        """
        # We need to join or subquery to get the status at the right time for all vehicles
        # This query finds the latest status <= request_time for each vehicle in the RSU
        query = """
            WITH RecentStatus AS (
                SELECT DISTINCT ON (vehicle_id) *
                FROM vehicle_status
                WHERE rsu_id = %s 
                  AND vehicle_id != %s
                  AND update_time <= %s
                ORDER BY vehicle_id, update_time DESC
            )
            SELECT *, 
                   SQRT(POWER(pos_x - %s, 2) + POWER(pos_y - %s, 2)) as dist_to_origin
            FROM RecentStatus
            ORDER BY dist_to_origin ASC, cpu_available DESC
            LIMIT %s
        """
        self.cursor.execute(query, (rsu_id, origin_vehicle_id, request_time, origin_x, origin_y, Config.MAX_NEIGHBORS))
        return self.cursor.fetchall()

    def _fetch_rsu_status(self, rsu_id, request_time):
        """Fetches the RSU status closest to the request time."""
        if rsu_id is None: return None
        
        # Try to match rsu_id (cast to string as DB uses VARCHAR)
        query = """
            SELECT * FROM rsu_status 
            WHERE rsu_id = %s AND update_time <= %s
            ORDER BY update_time DESC 
            LIMIT 1
        """
        self.cursor.execute(query, (str(rsu_id), request_time))
        return self.cursor.fetchone()

    def _fetch_rsu_metadata(self, rsu_id):
        """Fetches static metadata for the RSU."""
        if rsu_id is None: return None
        
        query = "SELECT * FROM rsu_metadata WHERE rsu_id = %s"
        self.cursor.execute(query, (str(rsu_id),))
        return self.cursor.fetchone()

    def _map_db_to_entities(self, request_row):
        """Maps DB rows to internal Entity objects for compatibility."""
        # 1. Create Task
        self.current_task = Task(
            task_id=request_row['task_id'],
            size=request_row['task_size_bytes'] / 1024 / 1024, # Convert to MB
            cpu_req=request_row['cpu_cycles'] / 1000000,       # Convert to Megacycles
            vehicle_id=request_row['vehicle_id'],
            deadline=request_row['deadline_seconds'],
            qos=request_row['qos_value'],
            created_time=request_row['request_time']
        )
        
        # 2. Create Origin Vehicle
        # Use the request time to fetch historical status
        v_status = self._fetch_vehicle_status(request_row['vehicle_id'], request_row['request_time'])
        
        if not v_status:
            # Fallback if status missing (shouldn't happen in sync sim)
            # Use data from request_row itself if available (it has snapshot)
            v_status = {
                'vehicle_id': request_row['vehicle_id'],
                'cpu_total': 5000, 'mem_total': 4096,
                'pos_x': request_row.get('pos_x', 0), 
                'pos_y': request_row.get('pos_y', 0), 
                'speed': request_row.get('speed', 0), 
                'heading': 0, 'acceleration': 0,
                'cpu_available': request_row.get('vehicle_cpu_available', 5000),
                'mem_available': request_row.get('vehicle_mem_available', 4096)
            }
            print(f"[DB-INFO] Using snapshot from request row for {request_row['vehicle_id']}")
            
        self.task_origin_vehicle = Vehicle(
            v_id=request_row['vehicle_id'],
            cpu_total=v_status.get('cpu_total', 5000),
            memory_total=v_status.get('mem_total', 4096)
        )
        # Update dynamic props
        self.task_origin_vehicle.pos_x = v_status.get('pos_x', 0)
        self.task_origin_vehicle.pos_y = v_status.get('pos_y', 0)
        self.task_origin_vehicle.speed = v_status.get('speed', 0)
        self.task_origin_vehicle.heading = v_status.get('heading', 0)
        self.task_origin_vehicle.cpu_avail = v_status.get('cpu_available', 0)
        self.task_origin_vehicle.memory_avail = v_status.get('mem_available', 0)
        self.task_origin_vehicle.cpu_utilization = v_status.get('cpu_utilization', 0.0)
        self.task_origin_vehicle.mem_utilization = v_status.get('mem_utilization', 0.0)
        self.task_origin_vehicle.processing_count = v_status.get('processing_count', 0)
        self.task_origin_vehicle.current_tasks = v_status.get('queue_length', 0)
        
        # 3. Create RSU
        rsu_id = request_row['rsu_id']
        
        # Fetch dynamic status and static metadata
        rsu_status = self._fetch_rsu_status(rsu_id, request_row['request_time'])
        rsu_meta = self._fetch_rsu_metadata(rsu_id)
        
        # Defaults
        cpu_total = 10000
        mem_total = 8192
        bandwidth = Config.BANDWIDTH_BASE
        pos_x = 0
        pos_y = 0
        
        if rsu_meta:
            pos_x = rsu_meta.get('pos_x', 0)
            pos_y = rsu_meta.get('pos_y', 0)
            bandwidth = rsu_meta.get('bandwidth_mbps', Config.BANDWIDTH_BASE)
            # If metadata has capacity, use it (assuming GHz -> *1000 for MHz approx, or just raw scale)
            # Adjust scaling based on your simulation units. Assuming 10000 is the baseline.
            if rsu_meta.get('cpu_capacity_ghz'): cpu_total = rsu_meta['cpu_capacity_ghz'] * 1000 
            if rsu_meta.get('memory_capacity_gb'): mem_total = rsu_meta['memory_capacity_gb'] * 1024 
            
        elif rsu_id is not None and isinstance(rsu_id, int) and rsu_id < len(Config.RSU_LOCATIONS):
             # Fallback to Config if metadata missing and ID is valid index
             pos_x, pos_y = Config.RSU_LOCATIONS[rsu_id]

        self.active_rsu = RSU(
            rsu_id=rsu_id,
            pos_x=pos_x,
            pos_y=pos_y,
            cpu_total=cpu_total,
            memory_total=mem_total,
            bandwidth=bandwidth
        )
        
        if rsu_status:
            self.active_rsu.cpu_avail = rsu_status.get('cpu_available', cpu_total)
            self.active_rsu.memory_avail = rsu_status.get('memory_available', mem_total)
            self.active_rsu.queue_length = rsu_status.get('queue_length', 0)
            self.active_rsu.cpu_utilization = rsu_status.get('cpu_utilization', 0.0)
            self.active_rsu.memory_utilization = rsu_status.get('memory_utilization', 0.0)
            self.active_rsu.processing_count = rsu_status.get('processing_count', 0)
            self.active_rsu.max_concurrent_tasks = rsu_status.get('max_concurrent_tasks', 10)
        else:
             print(f"[DB-WARN] No status found for RSU {rsu_id} at {request_row['request_time']}")
        
        # 4. Candidates
        # Pass origin location and time to sort by distance
        neighbor_rows = self._fetch_neighbors(
            rsu_id, 
            request_row['vehicle_id'], 
            self.task_origin_vehicle.pos_x, 
            self.task_origin_vehicle.pos_y,
            request_row['request_time']
        )
        
        self.candidates = []
        for row in neighbor_rows:
            v = Vehicle(row['vehicle_id'], row.get('cpu_total', 5000), row.get('mem_total', 4096))
            v.pos_x = row.get('pos_x', 0)
            v.pos_y = row.get('pos_y', 0)
            v.speed = row.get('speed', 0)
            v.heading = row.get('heading', 0)
            v.cpu_avail = row.get('cpu_available', 0)
            v.memory_avail = row.get('mem_available', 0)
            v.cpu_utilization = row.get('cpu_utilization', 0.0)
            v.mem_utilization = row.get('mem_utilization', 0.0)
            v.processing_count = row.get('processing_count', 0)
            v.current_tasks = row.get('queue_length', 0)
            self.candidates.append(v)
            
        if not self.candidates:
            print(f"[DB-INFO] No neighbors found for task {request_row['task_id']}")

    def reset(self):
        """
        Waits for the next request from the DB and sets up the environment.
        """
        request_row = self._fetch_latest_request()
        self._map_db_to_entities(request_row)
        return self._get_state()

    def _get_state(self):
        """
        Constructs state vector dynamically based on Config.DB_CONFIG['state_columns'].
        """
        state = []
        
        # 1. Task Features
        # We map the config column names to attributes of the Task object or raw DB row
        # For simplicity, we use the Entity objects we created, but we could use raw dicts
        task_cols = Config.DB_CONFIG["state_columns"]["task"]
        for col in task_cols:
            # Map 'size' -> self.current_task.size, etc.
            if hasattr(self.current_task, col):
                state.append(getattr(self.current_task, col))
            elif col == "cpu_req": state.append(self.current_task.cpu_req)
            else: state.append(0.0) # Default for missing
            
        # 2. RSU Features
        rsu_cols = Config.DB_CONFIG["state_columns"]["rsu"]
        for col in rsu_cols:
            if hasattr(self.active_rsu, col):
                state.append(getattr(self.active_rsu, col))
            elif col == "bandwidth": state.append(Config.BANDWIDTH_BASE) # Static fallback
            else: state.append(0.0)

        # 3. Neighbor Features
        veh_cols = Config.DB_CONFIG["state_columns"]["vehicle"]
        
        # Sort candidates by distance (same as DummyEnv)
        self.candidates.sort(key=lambda v: (v.pos_x - self.active_rsu.pos_x)**2 + (v.pos_y - self.active_rsu.pos_y)**2)
        
        # Take top K
        selected_candidates = self.candidates[:Config.MAX_NEIGHBORS]
        
        for v in selected_candidates:
            for col in veh_cols:
                # Handle relative features if needed, or raw
                # The user asked for dynamic updates. 
                # If we want relative position, we need to calculate it.
                val = 0.0
                if col == "pos_x": val = (v.pos_x - self.active_rsu.pos_x) / Config.RSU_RANGE # Relative X
                elif col == "pos_y": val = (v.pos_y - self.active_rsu.pos_y) / Config.RSU_RANGE # Relative Y
                elif hasattr(v, col): val = getattr(v, col)
                
                # Normalize common fields if possible (heuristic)
                if col == "speed": val /= Config.MAX_SPEED
                if col == "cpu_avail": val /= 5000.0
                if col == "memory_avail": val /= 4096.0
                if col == "processing_count": val /= 10.0
                # cpu_utilization and mem_utilization are already 0.0-1.0, no need to normalize
                
                state.append(val)
                
        # Padding if fewer neighbors than MAX_NEIGHBORS
        expected_len = Config.STATE_DIM
        current_len = len(state)
        if current_len < expected_len:
            state.extend([0.0] * (expected_len - current_len))
            
        return np.array(state, dtype=np.float32)

    def get_action_mask(self):
        # Reuse the logic from IoVDummyEnv since we mapped to Entities
        mask = np.zeros(Config.ACTION_DIM, dtype=np.float32)
        for i, v in enumerate(self.candidates[:Config.MAX_NEIGHBORS]):
            # Check if vehicle has enough resources (removed battery_avail check)
            if v.memory_avail > self.current_task.size and v.cpu_avail > 0:
                mask[i] = 1.0
        
        # 2. RSU (Infrastructure)
        # Check if RSU has enough resources and capacity
        if (self.active_rsu.cpu_avail > self.current_task.cpu_req * 0.1 and 
            self.active_rsu.memory_avail > self.current_task.size and
            self.active_rsu.processing_count < self.active_rsu.max_concurrent_tasks):
             mask[Config.MAX_NEIGHBORS] = 1.0 
        
        # Local action removed
        return mask

    def _insert_decision(self, action, decision_type, target_id=None):
        """
        Inserts the agent's decision into the offloading_decisions table.
        """
        query = """
            INSERT INTO offloading_decisions 
            (task_id, vehicle_id, rsu_id, decision_time, decision_type, target_service_vehicle_id, 
             confidence_score, estimated_completion_time, decision_reason, payload)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        # Use request time + small delta to simulate processing time
        decision_time = self.current_task.created_time + 0.05 
        
        payload = json.dumps({"agent": "DDQN", "action": int(action)})
        
        try:
            self.cursor.execute(query, (
                self.current_task.task_id,
                self.task_origin_vehicle.v_id,
                self.active_rsu.rsu_id,
                decision_time,
                decision_type,
                target_id,
                1.0, 
                0.0, 
                "RL_Agent_Decision",
                payload
            ))
            self.conn.commit()
            print(f"[DB] Inserted decision {decision_type} for task {self.current_task.task_id}")
        except Exception as e:
            print(f"[DB-ERROR] Failed to insert decision: {e}")
            self.conn.rollback()

    def _wait_for_completion(self, task_id, timeout=10.0):
        """
        Polls offloaded_task_completions for the result of the task.
        """
        start_time = time.time()
        poll_interval = 0.5
        
        while (time.time() - start_time) < timeout:
            query = "SELECT * FROM offloaded_task_completions WHERE task_id = %s"
            self.cursor.execute(query, (task_id,))
            row = self.cursor.fetchone()
            
            if row:
                return row
            
            time.sleep(poll_interval)
            
        print(f"[DB-WARN] Timeout waiting for completion of task {task_id}")
        return None

    def step(self, action):
        done = True
        
        # 1. Decode Action
        decision_type = "UNKNOWN"
        target_id = None
        
        if action < Config.MAX_NEIGHBORS:
            if action < len(self.candidates):
                decision_type = "SERVICE_VEHICLE"
                target_id = self.candidates[action].v_id
            else:
                # Invalid neighbor index
                return self._get_state(), Config.REWARD_FAILURE, True, {"latency": 10.0, "success": 0, "reason": "Invalid_Neighbor"}
                
        elif action == Config.MAX_NEIGHBORS:
            decision_type = "RSU"
            target_id = None # RSU is implicit or we could put RSU ID
            
        else:
            return self._get_state(), Config.REWARD_FAILURE, True, {"latency": 10.0, "success": 0, "reason": "Invalid_Action"}

        # 2. Insert Decision to DB
        self._insert_decision(action, decision_type, target_id)
        
        # 3. Wait for Simulator Result
        # We use a timeout from Config or default
        timeout = Config.DB_CONFIG.get("poll_timeout", 10.0)
        result = self._wait_for_completion(self.current_task.task_id, timeout)
        
        if not result:
            # Timeout or Error
            return self._get_state(), Config.REWARD_FAILURE, True, {"latency": timeout, "success": 0, "reason": "Timeout"}
            
        # 4. Calculate Reward from Real Data
        latency = result['total_latency']
        success = result['success']
        
        # Energy estimation (since DB might not have it)
        # We use the same physics model for energy as a proxy, or 0 if we don't care
        energy = 0.0
        if decision_type == "RSU":
            energy = self.current_task.cpu_req * 0.001
        else:
            energy = self.current_task.cpu_req * 0.002

        if success:
            rew_lat = Config.W_LATENCY * (1.0 - min(latency/2.0, 1.0))
            rew_ene = Config.W_ENERGY * (1.0 - min(energy/5.0, 1.0))
            rew_dead = Config.W_DEADLINE * 1.0
            reward = (rew_lat + rew_ene + rew_dead) * 10 * self.current_task.qos
        else:
            reward = Config.REWARD_FAILURE * self.current_task.qos

        # Normalize
        reward = reward / Config.REWARD_SCALE
        
        info = {
            "latency": latency,
            "energy": energy,
            "success": 1 if success else 0,
            "real_result": result
        }
        
        return self._get_state(), reward, done, info

    def close(self):
        if self.cursor: self.cursor.close()
        if self.conn: self.conn.close()