import numpy as np
import math
import random
import time
import types
import json
import os
import redis
from src.config import Config
from src.entities import Vehicle, Task, RSU
from src.local_env import get_env_int

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
        # 1. Collect ALL vehicles from all RSUs (multi-RSU enhancement)
        # Previously: only vehicles within active RSU range
        # Now: consider vehicles from ALL RSUs to enable global optimization
        all_candidates = []

        for v in self.vehicles:
            # Check if vehicle is within range of ANY RSU
            for rsu in self.rsus:
                dist_to_rsu = math.sqrt((v.pos_x - rsu.pos_x)**2 + (v.pos_y - rsu.pos_y)**2)
                if dist_to_rsu <= Config.RSU_RANGE:
                    all_candidates.append((v, rsu))  # (vehicle, its_serving_rsu)
                    break  # Each vehicle served by closest RSU only

        # 2. Sort by Euclidean distance to task-origin vehicle (global optimization)
        # Previously: sorted by distance to active RSU
        # Now: sorted by distance to the task's source vehicle
        self.candidates = sorted(
            all_candidates,
            key=lambda item: (
                (item[0].pos_x - self.task_origin_vehicle.pos_x)**2 +
                (item[0].pos_y - self.task_origin_vehicle.pos_y)**2
            )
        )[:Config.MAX_NEIGHBORS]  # Select top 12 closest (was 5)

        # Extract just the vehicles for feature extraction (RSU info already in state)
        candidate_vehicles = [v for v, rsu in self.candidates]

        # 3. Build State Vector
        state = []
        state.extend([self.current_task.size, self.current_task.cpu_req, self.current_task.deadline, self.current_task.qos])
        state.extend(self.active_rsu.to_feature_vector())

        # Use relative features relative to task-origin vehicle (not active RSU)
        for v in candidate_vehicles:
            state.extend(v.to_relative_feature_vector(self.task_origin_vehicle.pos_x, self.task_origin_vehicle.pos_y))

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
        Accounts for multi-RSU candidates where self.candidates contains (vehicle, rsu) tuples.
        """
        mask = np.zeros(Config.ACTION_DIM, dtype=np.float32)

        # 1. Vehicle Candidates (Neighbors selected from all RSUs)
        for i, item in enumerate(self.candidates):
            # Extract vehicle from (vehicle, rsu) tuple
            v = item[0] if isinstance(item, tuple) else item
            if v.battery_avail > 5.0 and v.memory_avail > self.current_task.size:
                mask[i] = 1.0

        # 2. RSU (Infrastructure) - always accessible
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
            # Extract vehicle from (vehicle, rsu) tuple
            target_item = self.candidates[action]
            target = target_item[0] if isinstance(target_item, tuple) else target_item

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


class IoVRedisEnv:
    """
    Redis-backed DRL environment for IoV task offloading.

    Action space (no local execution — the vehicle decides locally vs offload itself):
        [RSU_0, ..., RSU_{N-1}, SV_0, ..., SV_{K-1}]
        N = num_rsus from config,  K = max_neighbors from config

    State space (fully configurable via redis_config.json):
        [task_features | rsu_0_features | ... | rsu_{N-1}_features | sv_0_features | ... | sv_{K-1}_features]

    DATA FLOW:
        READS:  Redis  — task request queue, vehicle states, RSU resource states
        WRITES: Redis  — offloading decision for the simulator to pick up
    """

    def __init__(self, redis_db: int | None = None, instance_id: int = 0, tau_enabled: bool = True):
        redis_db = get_env_int("DEFAULT_REDIS_DB", 0) if redis_db is None else redis_db
        self.instance_id = instance_id
        self.r = redis.Redis(
            host=Config.REDIS_HOST,
            port=Config.REDIS_PORT,
            db=redis_db,
            decode_responses=True
        )
        print(f"✓ [DRL-{instance_id}] Redis connected (db={redis_db}): {self.r.ping()}")

        # State-vector configuration loaded from redis_config.json
        self.task_fields    = Config.REDIS_TASK_FIELDS
        self.rsu_fields     = Config.REDIS_RSU_FIELDS
        self.vehicle_fields = Config.REDIS_VEHICLE_FIELDS
        self.norm           = Config.REDIS_NORMALIZATION
        self.rsu_ids        = Config.RSU_IDS
        self.num_rsus       = Config.NUM_RSUS

        # Phase 2: SINR -> tau parameters (aligned with secondary DT defaults)
        self.dt2_run_id = Config.DT2_RUN_ID
        self.q_cycle_scan_limit = Config.DT2_Q_SCAN_LIMIT
        self.prediction_step_s = Config.DT2_PRED_STEP_S
        self.bandwidth_hz = Config.LINK_BANDWIDTH_HZ
        self.rate_efficiency = Config.LINK_RATE_EFFICIENCY
        self.default_output_ratio = Config.TASK_OUTPUT_RATIO
        self.default_tau_penalty_s = Config.TAU_MISSING_PENALTY_S
        self.tau_enabled = bool(tau_enabled)

        # Episode state
        self.task_request     = {}   # raw dict from Redis request hash
        self.rsu_states       = {}   # rsu_id -> state dict
        self.candidate_states = []   # raw state dicts (for _build_state)
        # Public SimpleNamespace objects consumed by baseline agents:
        self.rsus       = []   # one per RSU in the network
        self.candidates = []   # one per service-vehicle candidate
        self.active_rsu = None # SimpleNamespace of RSU that received the task

    # ------------------------------------------------------------------
    # DATA FETCHING
    # ------------------------------------------------------------------

    def _poll_request(self):
        """Blocking pop from the simulation's offloading request queue."""
        while True:
            result = self.r.blpop('offloading_requests:queue', timeout=1)
            if result:
                _, task_id = result
                data = self.r.hgetall(f"task:{task_id}:request")
                if data:
                    mem_footprint_bytes = float(data.get('mem_footprint_bytes', 0))
                    input_size_bytes = float(data.get('input_size_bytes', mem_footprint_bytes))
                    output_size_bytes = float(data.get('output_size_bytes', input_size_bytes * self.default_output_ratio))
                    cpu_cycles = float(data.get('cpu_cycles', 0))
                    return {
                        'task_id':         task_id,
                        'vehicle_id':      data.get('vehicle_id', ''),
                        'rsu_id':          data.get('rsu_id', self.rsu_ids[0]),
                        # Convert raw bytes → normalized units used in state vector
                        'mem_footprint_mb':  mem_footprint_bytes / (1024 * 1024),
                        'cpu_req_mcycles':   cpu_cycles / 1e6,
                        'deadline_s':        float(data.get('deadline_seconds', 1.0)),
                        'qos':               float(data.get('qos_value', 1.0)),
                        # Raw task metadata used for tau computation
                        'input_size_bytes':  input_size_bytes,
                        'output_size_bytes': output_size_bytes,
                        'cpu_cycles':        cpu_cycles,
                    }

    def _fetch_rsu_state(self, rsu_id):
        """Fetch RSU resource state from Redis (key: rsu:{rsu_id}:resources)."""
        data = self.r.hgetall(f"rsu:{rsu_id}:resources")
        if not data:
            return {}
        cpu_avail = float(data.get('cpu_available', 0))
        return {
            'cpu_available':    cpu_avail,
            'memory_available': float(data.get('memory_available', 0)),
            # 50 is the practical max queue depth at an RSU under typical urban IoV load
            'queue_length':     int(float(data.get('queue_length', 0))),
            'processing_count': int(float(data.get('processing_count', 0))),
            'cpu_utilization':  float(data.get('cpu_utilization', 0.0)),  # written directly by sim
            'pos_x':            float(data.get('pos_x', 0)),
            'pos_y':            float(data.get('pos_y', 0)),
            'sinr':             float(data.get('sinr', 0)),  # populated by simulator in future
        }

    def _fetch_vehicle_state(self, vehicle_id):
        """Fetch vehicle state from Redis (key: vehicle:{vehicle_id}:state)."""
        data = self.r.hgetall(f"vehicle:{vehicle_id}:state")
        if not data:
            return None
        return {
            'vehicle_id':       vehicle_id,
            'cpu_available':    float(data.get('cpu_available', 0)),
            'mem_available':    float(data.get('mem_available', 0)),
            'cpu_utilization':  float(data.get('cpu_utilization', 0)),
            'mem_utilization':  float(data.get('mem_utilization', 0)),
            'queue_length':     int(float(data.get('queue_length', 0))),
            'processing_count': int(float(data.get('processing_count', 0))),
            'speed':            float(data.get('speed', 0)),
            'heading':          float(data.get('heading', 0)),
            'acceleration':     float(data.get('acceleration', 0)),
            'pos_x':            float(data.get('pos_x', 0)),
            'pos_y':            float(data.get('pos_y', 0)),
            'sinr':             float(data.get('sinr', 0)),  # populated by simulator in future
            'distance_to_origin': 0.0,  # computed after origin is known
            # Phase 2 tau features (filled later for selected top-K candidates)
            'tau_up': 0.0,
            'tau_comp': 0.0,
            'tau_down': 0.0,
            'tau_total': 0.0,
        }

    def _get_latest_q_cycle(self):
        """Read latest secondary Q cycle index from Redis."""
        latest_key = f"dt2:q:{self.dt2_run_id}:latest"
        raw = self.r.hget(latest_key, "cycle_index")
        if raw is not None:
            try:
                return int(raw)
            except (TypeError, ValueError):
                pass

        stream_key = f"dt2:q:{self.dt2_run_id}:entries"
        tail = self.r.xrevrange(stream_key, count=1)
        if not tail:
            return None
        _, fields = tail[0]
        try:
            return int(fields.get('cycle_index', 0))
        except (TypeError, ValueError):
            return None

    def _process_q_entries(self, cycle_id):
        """Build per-link SINR sequences for one Q cycle."""
        stream_key = f"dt2:q:{self.dt2_run_id}:entries"
        entries = self.r.xrevrange(stream_key, count=self.q_cycle_scan_limit)
        links = {}
        seen_target_cycle = False

        for _entry_id, fields in entries:
            try:
                entry_cycle = int(fields.get('cycle_index', 0))
            except (TypeError, ValueError):
                continue

            if entry_cycle > cycle_id:
                continue
            if entry_cycle < cycle_id:
                if seen_target_cycle:
                    break
                continue
            seen_target_cycle = True

            link_type = fields.get('link_type', '')
            tx_id = fields.get('tx_id', '')
            rx_id = fields.get('rx_id', '')

            try:
                sinr_db = float(fields.get('sinr_db', -100.0))
            except (TypeError, ValueError):
                sinr_db = -100.0

            try:
                predicted_time = float(fields.get('predicted_time', 0.0))
            except (TypeError, ValueError):
                predicted_time = 0.0

            link_key = f"{link_type}:{tx_id}:{rx_id}"
            if link_key not in links:
                links[link_key] = {
                    'link_type': link_type,
                    'tx_id': tx_id,
                    'rx_id': rx_id,
                    'sinr_sequence': [],
                    'predicted_times': [],
                }

            links[link_key]['sinr_sequence'].append(sinr_db)
            links[link_key]['predicted_times'].append(predicted_time)

        # Stream is newest-first; restore temporal order per link.
        for link in links.values():
            combined = sorted(zip(link['predicted_times'], link['sinr_sequence']))
            link['predicted_times'] = [ts for ts, _ in combined]
            link['sinr_sequence'] = [sinr for _, sinr in combined]

        return links

    def _sinr_db_to_rate_bps(self, sinr_db):
        """Shannon-like link-rate model aligned with external controller."""
        sinr_linear = 10.0 ** (sinr_db / 10.0)
        sinr_linear = max(sinr_linear, 1e-9)
        return self.bandwidth_hz * math.log2(1.0 + sinr_linear) * self.rate_efficiency

    def _accumulate_transmission_time(self, sinr_sequence, data_bits):
        """Accumulate slot capacity until data_bits are transmitted."""
        if data_bits <= 0:
            return 0.0
        if not sinr_sequence:
            return self.default_tau_penalty_s

        remaining = data_bits
        elapsed_s = 0.0

        for sinr_db in sinr_sequence:
            rate_bps = max(self._sinr_db_to_rate_bps(sinr_db), 1.0)
            bits_this_slot = rate_bps * self.prediction_step_s

            if remaining <= bits_this_slot:
                elapsed_s += remaining / rate_bps
                return elapsed_s

            remaining -= bits_this_slot
            elapsed_s += self.prediction_step_s

        # If the window is too short, continue with the last observed rate.
        tail_rate_bps = max(self._sinr_db_to_rate_bps(sinr_sequence[-1]), 1.0)
        elapsed_s += remaining / tail_rate_bps
        return elapsed_s

    def _slice_from_offset(self, sinr_sequence, predicted_times, t_start):
        """Slice DL SINR from t0 + t_start to model tau_up + tau_comp delay."""
        if not predicted_times:
            return sinr_sequence

        t0 = predicted_times[0]
        sliced = [
            sinr
            for predicted_time, sinr in zip(predicted_times, sinr_sequence)
            if predicted_time >= t0 + t_start
        ]
        return sliced if sliced else sinr_sequence[-1:]

    def _get_compute_capacity_hz(self, target_id):
        """Read target compute capacity in Hz from RSU/vehicle resource keys."""
        rsu = self.rsu_states.get(target_id)
        if rsu:
            cap = float(rsu.get('cpu_available', 0.0))
            if cap > 0:
                return cap

        vehicle_state = self._fetch_vehicle_state(target_id)
        if vehicle_state:
            cap = float(vehicle_state.get('cpu_available', 0.0))
            if cap > 0:
                return cap

        # Safe fallback to avoid divide-by-zero.
        return 1e9

    def _compute_v2v_tau(self, source_vehicle, service_vehicle, links, d_in_bits, d_out_bits, c_cycles):
        """Compute tau tuple for one selected service vehicle candidate."""
        ul_key = f"V2V:{source_vehicle}:{service_vehicle}"
        dl_key = f"V2V:{service_vehicle}:{source_vehicle}"
        ul_data = links.get(ul_key)
        dl_data = links.get(dl_key)

        if not ul_data or not dl_data:
            return {
                'tau_up': self.default_tau_penalty_s,
                'tau_comp': self.default_tau_penalty_s,
                'tau_down': self.default_tau_penalty_s,
                'tau_total': self.default_tau_penalty_s * 3.0,
            }

        tau_up = self._accumulate_transmission_time(ul_data['sinr_sequence'], d_in_bits)
        f_avail_hz = self._get_compute_capacity_hz(service_vehicle)
        tau_comp = c_cycles / max(f_avail_hz, 1.0)
        dl_sinr_seq = self._slice_from_offset(
            dl_data['sinr_sequence'],
            dl_data['predicted_times'],
            tau_up + tau_comp,
        )
        tau_down = self._accumulate_transmission_time(dl_sinr_seq, d_out_bits)
        tau_total = tau_up + tau_comp + tau_down

        return {
            'tau_up': tau_up,
            'tau_comp': tau_comp,
            'tau_down': tau_down,
            'tau_total': tau_total,
        }

    def _attach_tau_to_candidates(self, source_vehicle, states):
        """Phase 2 core: compute tau for already-selected top-K closest candidates."""
        if not states:
            return states

        if not self.tau_enabled:
            for state in states:
                state['tau_up'] = 0.0
                state['tau_comp'] = 0.0
                state['tau_down'] = 0.0
                state['tau_total'] = 0.0
            return states

        cycle_id = self._get_latest_q_cycle()
        if cycle_id is None:
            for state in states:
                state['tau_up'] = self.default_tau_penalty_s
                state['tau_comp'] = self.default_tau_penalty_s
                state['tau_down'] = self.default_tau_penalty_s
                state['tau_total'] = self.default_tau_penalty_s * 3.0
            return states

        links = self._process_q_entries(cycle_id)
        if not links:
            for state in states:
                state['tau_up'] = self.default_tau_penalty_s
                state['tau_comp'] = self.default_tau_penalty_s
                state['tau_down'] = self.default_tau_penalty_s
                state['tau_total'] = self.default_tau_penalty_s * 3.0
            return states

        d_in_bits = max(0.0, float(self.task_request.get('input_size_bytes', 0.0)) * 8.0)
        d_out_bits = max(0.0, float(self.task_request.get('output_size_bytes', 0.0)) * 8.0)
        c_cycles = max(0.0, float(self.task_request.get('cpu_cycles', 0.0)))

        # Fallback for tasks where fields may not yet be populated.
        if d_in_bits <= 0.0:
            d_in_bits = max(0.0, float(self.task_request.get('mem_footprint_mb', 0.0)) * 1024.0 * 1024.0 * 8.0)
        if d_out_bits <= 0.0:
            d_out_bits = d_in_bits * self.default_output_ratio
        if c_cycles <= 0.0:
            c_cycles = max(0.0, float(self.task_request.get('cpu_req_mcycles', 0.0)) * 1e6)

        for state in states:
            target_id = state['vehicle_id']
            tau = self._compute_v2v_tau(source_vehicle, target_id, links, d_in_bits, d_out_bits, c_cycles)
            state.update(tau)

        return states

    # Maximum V2V communication range in metres.
    # Matches the ~178 m free-space threshold derived from PayloadVehicleApp's
    # estimateV2vRssiDbm() at the -85 dBm sensitivity limit.  We use a slightly
    # larger value (250 m) to give the agent a small safety margin for vehicles
    # that are momentarily near the boundary.
    MAX_SV_DISTANCE_M = 250.0

    # Maximum V2V communication range in metres.
    # Matches the ~178 m free-space threshold derived from PayloadVehicleApp's
    # estimateV2vRssiDbm() at the -85 dBm sensitivity limit. Keep the Python
    # mask near the simulator's real V2V range so DDQN does not learn risky
    # service-vehicle actions that later fail as SV_OUT_OF_RANGE/fallback.
    MAX_SV_DISTANCE_M = 180.0

    def _fetch_candidates(self, origin_id, origin_x, origin_y):
        """
        Fetch service-vehicle candidates from the Redis sorted set.

        Changes vs. original:
        1. Fetch a much larger pool (50) so nearby SVs ranked lower by CPU
           score are not silently excluded.
        2. Prune stale entries whose vehicle:{id}:state key has expired.
        3. Filter to only SVs within MAX_SV_DISTANCE_M of the origin vehicle.
        4. Re-sort the survivors by distance and keep the MAX_NEIGHBORS closest.
        """
        top = self.r.zrevrange('service_vehicles:available', 0, -1, withscores=True)
        candidates, states = [], []
        stale_ids = []
        for vehicle_id, _ in top:
            if vehicle_id == origin_id:
                continue
            state = self._fetch_vehicle_state(vehicle_id)
            if state is None:
                # Vehicle state has expired — remove ghost entry from sorted set
                stale_ids.append(vehicle_id)
                continue
            dist = math.sqrt(
                (state['pos_x'] - origin_x) ** 2 + (state['pos_y'] - origin_y) ** 2
            )
            state['distance_to_origin'] = dist
            # Only consider SVs within V2V communication range
            if dist <= self.MAX_SV_DISTANCE_M:
                candidates.append(vehicle_id)
                states.append(state)

        # Clean up stale sorted-set entries in one pipeline call
        if stale_ids:
            pipe = self.r.pipeline(transaction=False)
            for sid in stale_ids:
                pipe.zrem('service_vehicles:available', sid)
            pipe.execute()

        # Sort closest first, keep at most MAX_NEIGHBORS
        paired = sorted(zip(candidates, states), key=lambda x: x[1]['distance_to_origin'])
        paired = paired[:Config.MAX_NEIGHBORS]
        if paired:
            paired = paired[:Config.MAX_NEIGHBORS]
            candidates, states = zip(*paired)
            states = self._attach_tau_to_candidates(origin_id, list(states))
            return list(candidates), list(states)
        return [], []

    # ------------------------------------------------------------------
    # STATE VECTOR
    # ------------------------------------------------------------------

    def _extract_features(self, data_dict, fields, norm_section):
        """Normalize and extract a list of scalar features from a dict."""
        feats = []
        for field in fields:
            val  = float(data_dict.get(field, 0.0))
            norm = float(norm_section.get(field, 1.0))
            feats.append(val / norm if norm != 0 else 0.0)
        return feats

    def _build_state(self):
        """Build the full state vector from the current episode data."""
        state = []

        # --- Task features ---
        state.extend(self._extract_features(
            self.task_request, self.task_fields, self.norm.get('task', {})))

        # --- RSU features (one block per RSU in the network) ---
        for rsu_id in self.rsu_ids:
            rsu_state = self.rsu_states.get(rsu_id, {})
            state.extend(self._extract_features(
                rsu_state, self.rsu_fields, self.norm.get('rsu', {})))

        # --- Service-vehicle features (padded to MAX_NEIGHBORS) ---
        for i in range(Config.MAX_NEIGHBORS):
            if i < len(self.candidate_states):
                state.extend(self._extract_features(
                    self.candidate_states[i], self.vehicle_fields, self.norm.get('vehicle', {})))
            else:
                state.extend([0.0] * len(self.vehicle_fields))

        return np.array(state, dtype=np.float32)

    # ------------------------------------------------------------------
    # GYM-STYLE INTERFACE
    # ------------------------------------------------------------------

    def reset(self):
        """Wait for a new offloading request then build the initial state."""
        request = self._poll_request()
        print(f"\n[Redis-ENV] New Request: task={request['task_id']} vehicle={request['vehicle_id']}")

        self.task_request = request
        active_rsu_id     = request.get('rsu_id', self.rsu_ids[0])

        # Fetch all RSU states
        self.rsu_states = {rsu_id: self._fetch_rsu_state(rsu_id) for rsu_id in self.rsu_ids}

        # Build SimpleNamespace for each RSU (consumed by baseline agents)
        self.rsus = [
            types.SimpleNamespace(
                rsu_id       = rsu_id,
                cpu_avail    = self.rsu_states.get(rsu_id, {}).get('cpu_available', 0.0),
                queue_length = self.rsu_states.get(rsu_id, {}).get('queue_length', 0),
                pos_x        = self.rsu_states.get(rsu_id, {}).get('pos_x', 0.0),
                pos_y        = self.rsu_states.get(rsu_id, {}).get('pos_y', 0.0),
            )
            for rsu_id in self.rsu_ids
        ]
        self.active_rsu = next(
            (r for r in self.rsus if r.rsu_id == active_rsu_id), self.rsus[0]
        )

        # Fetch origin vehicle state for position reference
        origin_state = self._fetch_vehicle_state(request['vehicle_id'])
        if origin_state is None:
            print(f"[Redis-WARN] Origin vehicle {request['vehicle_id']} not found, retrying...")
            return self.reset()

        # Fetch candidate service vehicles → raw dicts + SimpleNamespace objects
        raw_ids, self.candidate_states = self._fetch_candidates(
            request['vehicle_id'], origin_state['pos_x'], origin_state['pos_y']
        )
        self.candidates = [
            types.SimpleNamespace(
                vehicle_id        = s['vehicle_id'],
                cpu_avail         = s.get('cpu_available', 0.0),
                mem_avail         = s.get('mem_available', 0.0),
                queue_length      = s.get('queue_length', 0),
                pos_x             = s.get('pos_x', 0.0),
                pos_y             = s.get('pos_y', 0.0),
                speed             = s.get('speed', 0.0),
                heading           = s.get('heading', 0.0),
                distance_to_origin= s.get('distance_to_origin', 9999.0),
                tau_up       = s.get('tau_up', 0.0),
                tau_comp     = s.get('tau_comp', 0.0),
                tau_down     = s.get('tau_down', 0.0),
                tau_total    = s.get('tau_total', 0.0),
            )
            for s in self.candidate_states
        ]

        return self._build_state()

    def get_action_mask(self):
        """
        Binary mask of size ACTION_DIM = NUM_RSUS + MAX_NEIGHBORS.
        Layout: [RSU_0, ..., RSU_{N-1}, SV_0, ..., SV_{K-1}]
        """
        mask = np.zeros(Config.ACTION_DIM, dtype=np.float32)

        # RSU actions — valid if RSU has available CPU
        for i, rsu_id in enumerate(self.rsu_ids):
            if float(self.rsu_states.get(rsu_id, {}).get('cpu_available', 0)) > 0:
                mask[i] = 1.0

        # Service-vehicle actions — valid only if candidate is within V2V range
        for j, sv in enumerate(self.candidates):
            dist = getattr(sv, 'distance_to_origin', 9999.0)
            if dist <= self.MAX_SV_DISTANCE_M:
                mask[self.num_rsus + j] = 1.0

        # Safety fallback: allow all RSUs if nothing else is valid
        if mask.sum() == 0:
            mask[:self.num_rsus] = 1.0

        return mask

    def step(self, action):
        """Single-agent step (thin wrapper over step_multi for backward compat)."""
        _, results = self.step_multi({'ddqn': action})
        reward, info = results['ddqn']
        return self._build_state(), reward, True, info

    def step_multi(self, actions: dict):
        """
        Execute decisions for ALL agents on the SAME task atomically.

        actions: {'ddqn': int, 'random': int, ...}
        Returns: (next_state, results_dict)
            results_dict: {agent_name: (reward, info)}
        """
        # Write decisions via shared validator (Phase 3).
        task_id = self.task_request['task_id']
        agent_decisions = self.write_decisions(task_id, actions)

        print(f"[Redis-ENV] Decisions written: { {a: t for a, (_, t) in agent_decisions.items()} }")

        # Wait for per-agent results from simulator
        results_raw = self._wait_for_multi_results(list(actions.keys()))

        # Calculate reward for each agent
        results = {}
        for agent, (dtype, tid) in agent_decisions.items():
            result   = results_raw.get(agent, {'status': 'FAILED', 'total_latency': 999, 'energy': 0.0})
            reward, info = self._calculate_reward(result, dtype, tid)
            results[agent] = (reward, info)

        return self._build_state(), results

    # ------------------------------------------------------------------
    # DECISION & RESULT
    # ------------------------------------------------------------------

    def _action_to_target(self, action):
        """Map action index → (decision_type, target_id)."""
        if action < self.num_rsus:
            return "RSU", self.rsu_ids[action]
        sv_idx = action - self.num_rsus
        if sv_idx < len(self.candidates):
            return "SERVICE_VEHICLE", self.candidates[sv_idx].vehicle_id
        # Fallback
        return "RSU", self.rsu_ids[0]

    def _wait_for_multi_results(self, agent_names: list):
        """
        Poll task:{id}:results until all agents' status fields are present.
        Returns a dict: agent_name → {'status', 'total_latency', 'energy'}.
        """
        task_id        = self.task_request['task_id']
        required       = {f'{a}_status' for a in agent_names}
        deadline       = time.time() + Config.REDIS_RESULT_TIMEOUT

        while time.time() < deadline:
            multi_data = self.r.hgetall(f"task:{task_id}:results")
            single_data = self.r.hgetall(f"task:{task_id}:result")
            data = self._normalize_result_data(multi_data, single_data)
            if required.issubset(data.keys()):
                return {
                    agent: {
                        'status':        data.get(f'{agent}_status', 'FAILED'),
                        'total_latency': float(data.get(f'{agent}_latency', 999)),
                        'energy':        float(data.get(f'{agent}_energy',  0.0)),
                    }
                    for agent in agent_names
                }
            time.sleep(Config.REDIS_POLL_INTERVAL)

        print(f"[Redis-WARN] Timeout waiting for results of task {task_id}")
        return {
            a: {'status': 'FAILED', 'total_latency': 999, 'energy': 0.0}
            for a in agent_names
        }

    def _calculate_reward(self, result, decision_type, target_id):
        """Compute reward from the task execution result."""
        success = result['status'] == 'COMPLETED_ON_TIME'
        latency = result['total_latency']
        energy  = result['energy']   # 0.0 dummy until simulator reports it

        if success:
            rew_lat  = Config.W_LATENCY  * (1.0 - min(latency / max(self.task_request['deadline_s'], 1e-6), 1.0))
            rew_ene  = Config.W_ENERGY   * (1.0 - min(energy / 5.0, 1.0))
            rew_dead = Config.W_DEADLINE * 1.0
            reward   = (rew_lat + rew_ene + rew_dead) * Config.REWARD_SCALE * self.task_request['qos']
        else:
            reward = Config.REWARD_FAILURE * self.task_request['qos']

        reward /= Config.REWARD_SCALE

        info = {
            'latency':       latency,
            'energy':        energy,
            'success':       1 if success else 0,
            'decision_type': decision_type,
            'target_id':     target_id,
        }
        return reward, info

    # ------------------------------------------------------------------
    # ASYNC HELPERS (used by the non-blocking training loop in main.py)
    # ------------------------------------------------------------------

    def _poll_request_nonblocking(self):
        """
        Non-blocking task fetch: returns a request dict or None if the queue is empty.
        Uses lpop (returns immediately) instead of blpop (blocks until data arrives).
        """
        task_id = self.r.lpop('offloading_requests:queue')
        if not task_id:
            return None
        data = self.r.hgetall(f"task:{task_id}:request")
        if not data:
            return None
        mem_footprint_bytes = float(data.get('mem_footprint_bytes', 0))
        input_size_bytes = float(data.get('input_size_bytes', mem_footprint_bytes))
        output_size_bytes = float(data.get('output_size_bytes', input_size_bytes * self.default_output_ratio))
        cpu_cycles = float(data.get('cpu_cycles', 0))
        return {
            'task_id':          task_id,
            'vehicle_id':       data.get('vehicle_id', ''),
            'rsu_id':           data.get('rsu_id', self.rsu_ids[0]),
            'mem_footprint_mb': mem_footprint_bytes / (1024 * 1024),
            'cpu_req_mcycles':  cpu_cycles / 1e6,
            'deadline_s':       float(data.get('deadline_seconds', 1.0)),
            'qos':              float(data.get('qos_value', 1.0)),
            'input_size_bytes': input_size_bytes,
            'output_size_bytes': output_size_bytes,
            'cpu_cycles':       cpu_cycles,
            'task_type':        data.get('task_type', 'UNKNOWN'),  # for per-type TensorBoard panels
        }

    def setup_from_request(self, request):
        """
        Populate env state from a pre-fetched request dict without blocking.
        Returns the state vector, or None if the vehicle state is not yet in Redis.
        Equivalent to reset() but works with an already-fetched request.
        """
        self.task_request = request
        self.task_type = request.get("task_type", "UNKNOWN")  # for per-type TensorBoard metrics
        active_rsu_id     = request.get('rsu_id', self.rsu_ids[0])

        self.rsu_states = {rsu_id: self._fetch_rsu_state(rsu_id) for rsu_id in self.rsu_ids}
        self.rsus = [
            types.SimpleNamespace(
                rsu_id       = rsu_id,
                cpu_avail    = self.rsu_states.get(rsu_id, {}).get('cpu_available', 0.0),
                queue_length = self.rsu_states.get(rsu_id, {}).get('queue_length', 0),
                pos_x        = self.rsu_states.get(rsu_id, {}).get('pos_x', 0.0),
                pos_y        = self.rsu_states.get(rsu_id, {}).get('pos_y', 0.0),
            )
            for rsu_id in self.rsu_ids
        ]
        self.active_rsu = next(
            (r for r in self.rsus if r.rsu_id == active_rsu_id), self.rsus[0]
        )

        origin_state = self._fetch_vehicle_state(request['vehicle_id'])
        if origin_state is None:
            return None  # vehicle state not yet in Redis — caller should skip this task

        raw_ids, self.candidate_states = self._fetch_candidates(
            request['vehicle_id'], origin_state['pos_x'], origin_state['pos_y']
        )
        self.candidates = [
            types.SimpleNamespace(
                vehicle_id        = s['vehicle_id'],
                cpu_avail         = s.get('cpu_available', 0.0),
                mem_avail         = s.get('mem_available', 0.0),
                queue_length      = s.get('queue_length', 0),
                pos_x             = s.get('pos_x', 0.0),
                pos_y             = s.get('pos_y', 0.0),
                speed             = s.get('speed', 0.0),
                heading           = s.get('heading', 0.0),
                tau_up       = s.get('tau_up', 0.0),
                tau_comp     = s.get('tau_comp', 0.0),
                tau_down     = s.get('tau_down', 0.0),
                tau_total    = s.get('tau_total', 0.0),
                distance_to_origin= s.get('distance_to_origin', 9999.0),
            )
            for s in self.candidate_states
        ]
        return self._build_state()

    def write_decisions(self, task_id, actions, trace_metadata=None):
        """
        Write all agent decisions to Redis atomically and return agent_decisions mapping.
        Extracted from step_multi() so it can be called without blocking on results.
        Returns: {agent_name: (decision_type, target_id)}
        """
        current_task_id = self.task_request.get('task_id')
        if current_task_id is not None and str(current_task_id) != str(task_id):
            raise ValueError(
                f"Task mismatch while writing decisions: request={current_task_id}, write={task_id}"
            )

        candidate_ids = {str(c.vehicle_id) for c in self.candidates}

        agent_decisions = {
            agent: self._action_to_target(act) for agent, act in actions.items()
        }

        # Phase 3 guard: ensure SERVICE_VEHICLE targets are always from this task's
        # top-K tau candidate list. If not, coerce to a safe fallback.
        for agent, (dtype, tid) in list(agent_decisions.items()):
            if dtype == "SERVICE_VEHICLE" and str(tid) not in candidate_ids:
                if self.candidates:
                    agent_decisions[agent] = ("SERVICE_VEHICLE", self.candidates[0].vehicle_id)
                else:
                    agent_decisions[agent] = ("RSU", self.rsu_ids[0])

        mapping = {'agents': ','.join(actions.keys())}
        for agent, (dtype, tid) in agent_decisions.items():
            mapping[f'{agent}_type']   = dtype
            mapping[f'{agent}_target'] = tid

        # Persist validation breadcrumbs for debugging and auditability.
        ddqn_target = agent_decisions.get('ddqn', ('RSU', self.rsu_ids[0]))[1]
        mapping['ddqn_candidate_in_topk'] = '1' if str(ddqn_target) in candidate_ids else '0'
        mapping['candidate_pool'] = ','.join(sorted(candidate_ids))

        pipe = self.r.pipeline()
        pipe.hset(f"task:{task_id}:decisions", mapping=mapping)
        pipe.expire(f"task:{task_id}:decisions", 300)
        ddqn_type, ddqn_target = agent_decisions.get('ddqn', ('RSU', self.rsu_ids[0]))
        single_mapping = {
            'agent': 'ddqn',
            'type': str(ddqn_type),
            'target': str(ddqn_target),
        }
        if trace_metadata:
            for k, v in trace_metadata.items():
                if v is None:
                    continue
                single_mapping[str(k)] = str(v)

        pipe.hset(f"task:{task_id}:decision", mapping=single_mapping)
        pipe.expire(f"task:{task_id}:decision", 300)
        pipe.execute()
        return agent_decisions

    @staticmethod
    def _normalize_result_data(multi_data, single_data):
        """
        Normalize simulator results from either multi-agent key (task:{id}:results)
        or single-agent key (task:{id}:result) into a common dict shape.
        """
        if multi_data and 'ddqn_status' in multi_data:
            return multi_data

        normalized = {}
        if single_data and ('status' in single_data or 'ddqn_status' in single_data):
            status = single_data.get('ddqn_status', single_data.get('status', 'FAILED'))
            latency = single_data.get('ddqn_latency', single_data.get('total_latency', '999'))
            energy = single_data.get('ddqn_energy', single_data.get('energy', '0.0'))
            reason = single_data.get('ddqn_reason', single_data.get('fail_reason', 'UNKNOWN'))
            normalized['ddqn_status'] = status
            normalized['ddqn_latency'] = latency
            normalized['ddqn_energy'] = energy
            normalized['ddqn_reason'] = reason
        return normalized

    def check_results_nonblocking(self, task_id, agent_names):
        """
        Non-blocking check: returns per-agent result dict once ddqn has reported.
        Missing baseline agents get a TIMEOUT placeholder so the training loop
        can still use the ddqn result for DDQN training.
        Returns None only if ddqn hasn't reported yet.
        """
        multi_data = self.r.hgetall(f"task:{task_id}:results")
        single_data = self.r.hgetall(f"task:{task_id}:result")
        data = self._normalize_result_data(multi_data, single_data)
        # Must have at least ddqn results to proceed
        if 'ddqn_status' not in data:
            return None
        return {
            agent: {
                'status':        data.get(f'{agent}_status', 'FAILED'),
                'total_latency': float(data.get(f'{agent}_latency', 999)),
                'energy':        float(data.get(f'{agent}_energy',  0.0)),
                'fail_reason':   data.get(f'{agent}_reason',
                                          'TIMEOUT' if f'{agent}_status' not in data else 'UNKNOWN'),
            }
            for agent in agent_names
        }

    def check_late_baselines(self, task_id, missing_agents):
        """
        Poll Redis for specific missing baseline agents.
        Returns (arrived_dict, still_missing_list).
        arrived_dict: {agent_name: result_dict} for agents that have now reported.
        """
        data = self.r.hgetall(f"task:{task_id}:results")
        arrived, still_missing = {}, []
        for agent in missing_agents:
            if f'{agent}_status' in data:
                arrived[agent] = {
                    'status':        data[f'{agent}_status'],
                    'total_latency': float(data.get(f'{agent}_latency', 999)),
                    'energy':        float(data.get(f'{agent}_energy', 0.0)),
                    'fail_reason':   data.get(f'{agent}_reason',
                                              'NONE' if data[f'{agent}_status'] == 'COMPLETED_ON_TIME'
                                              else 'UNKNOWN'),
                }
            else:
                still_missing.append(agent)
        return arrived, still_missing

    def batch_check_results(self, pending_dict):
        """
        Batch poll Redis for all pending tasks in a single pipeline round-trip.
        Returns a dict: {task_id: results_raw} for tasks whose ddqn_status has arrived.
        Tasks not yet ready are omitted from the returned dict.
        pending_dict: {task_id: entry} where entry has 'actions' key.
        """
        task_ids = list(pending_dict.keys())
        if not task_ids:
            return {}
        pipe = self.r.pipeline(transaction=False)
        for task_id in task_ids:
            pipe.hgetall(f"task:{task_id}:results")
            pipe.hgetall(f"task:{task_id}:result")
        all_data = pipe.execute()  # single network round-trip

        ready = {}
        for idx, task_id in enumerate(task_ids):
            multi_data = all_data[2 * idx]
            single_data = all_data[2 * idx + 1]
            data = self._normalize_result_data(multi_data, single_data)
            if not data or 'ddqn_status' not in data:
                continue
            agent_names = list(pending_dict[task_id]['actions'].keys())
            ready[task_id] = {
                agent: {
                    'status':        data.get(f'{agent}_status', 'FAILED'),
                    'total_latency': float(data.get(f'{agent}_latency', 999)),
                    'energy':        float(data.get(f'{agent}_energy',  0.0)),
                    'fail_reason':   data.get(f'{agent}_reason',
                                              'TIMEOUT' if f'{agent}_status' not in data else 'UNKNOWN'),
                }
                for agent in agent_names
            }
        return ready

    def batch_check_late_baselines(self, late_baselines_dict):
        """
        Batch poll Redis for late-arriving baselines in a single pipeline round-trip.
        Returns {task_id: (arrived_dict, still_missing_list)}.
        late_baselines_dict: {task_id: {'missing': [...], ...}}
        """
        task_ids = list(late_baselines_dict.keys())
        if not task_ids:
            return {}
        pipe = self.r.pipeline(transaction=False)
        for task_id in task_ids:
            pipe.hgetall(f"task:{task_id}:results")
        all_data = pipe.execute()

        results = {}
        for task_id, data in zip(task_ids, all_data):
            missing_agents = late_baselines_dict[task_id]['missing']
            arrived, still_missing = {}, []
            for agent in missing_agents:
                if data and f'{agent}_status' in data:
                    arrived[agent] = {
                        'status':        data[f'{agent}_status'],
                        'total_latency': float(data.get(f'{agent}_latency', 999)),
                        'energy':        float(data.get(f'{agent}_energy', 0.0)),
                        'fail_reason':   data.get(f'{agent}_reason',
                                                  'NONE' if data[f'{agent}_status'] == 'COMPLETED_ON_TIME'
                                                  else 'UNKNOWN'),
                    }
                else:
                    still_missing.append(agent)
            results[task_id] = (arrived, still_missing)
        return results

    def compute_reward_for(self, task_request, result, decision_type, target_id):
        """
        Standalone reward calculation given a stored task_request dict and an agent's result.
        Used by the async loop so it can compute rewards for any pending task independent
        of the current self.task_request state.
        result keys (from batch_check_single_results): status, latency, energy, reason
        """
        success = result['status'] == 'COMPLETED_ON_TIME'
        latency = result['latency']   # key written by writeSingleResult → batch_check_single_results
        energy  = result['energy']

        if success:
            rew_lat  = Config.W_LATENCY  * (1.0 - min(latency / max(task_request['deadline_s'], 1e-6), 1.0))
            rew_ene  = Config.W_ENERGY   * (1.0 - min(energy / 5.0, 1.0))
            rew_dead = Config.W_DEADLINE * 1.0
            reward   = (rew_lat + rew_ene + rew_dead) * Config.REWARD_SCALE * task_request['qos']
        else:
            reward = Config.REWARD_FAILURE * task_request['qos']

        reward /= Config.REWARD_SCALE
        return reward, {
            'latency':       latency,
            'energy':        energy,
            'success':       1 if success else 0,
            'fail_reason':   result.get('reason', 'NONE' if success else 'UNKNOWN'),
            'decision_type': decision_type,
            'target_id':     target_id,
        }

    # ── Single-agent API ──────────────────────────────────────────────────────

    def write_decision(self, task_id: str, action: int, agent_name: str) -> dict:
        """
        Write a single-agent decision to task:{task_id}:decision.
        Returns (decision_type, target_id) as a dict.
        """
        decision_type, target_id = self._action_to_target(action)
        pipe = self.r.pipeline()
        pipe.hset(f"task:{task_id}:decision", mapping={
            "agent":  agent_name,
            "type":   decision_type,
            "target": target_id,
        })
        # Write to the plural format that C++ (MyRSUApp) expects!
        pipe.hset(f"task:{task_id}:decisions", mapping={
            "agents": agent_name,
            f"{agent_name}_type": decision_type,
            f"{agent_name}_target": target_id,
        })
        pipe.expire(f"task:{task_id}:decision", 300)
        pipe.expire(f"task:{task_id}:decisions", 300)
        # Notify the Metrics Engine runner when it is active
        if self.r.get("engine_active") in ("1", "true", "True", "yes"):
            pipe.rpush("engine_requests:queue", task_id)
        pipe.execute()
        return {"type": decision_type, "target": target_id}

    def batch_check_single_results(self, pending: dict) -> dict:
        """
        Pipeline batch-read task:{id}:result for all pending task_ids.
        Returns {task_id: result_dict} for tasks whose result has arrived.
        result_dict keys: status, latency, energy, reason
        pending: {task_id: any}
        """
        task_ids = list(pending.keys())
        if not task_ids:
            return {}
        pipe = self.r.pipeline(transaction=False)
        for tid in task_ids:
            pipe.hgetall(f"task:{tid}:result")
        all_data = pipe.execute()
        ready = {}
        for tid, data in zip(task_ids, all_data):
            if data and "status" in data:
                # Check if there is an agent-specific overwrite (e.g., from DDQN_PENALTY)
                final_status = data.get("ddqn_status", data["status"])
                final_reason = data.get("ddqn_reason", data.get("reason", "UNKNOWN"))
                
                # If it fell back, ddqn_latency might be forced artificially high in C++
                final_lat = float(data.get("ddqn_latency", data.get("latency", 999.0)))
                final_ene = float(data.get("ddqn_energy", data.get("energy", 0.0)))
                
                ready[tid] = {
                    "status":  final_status,
                    "latency": final_lat,
                    "energy":  final_ene,
                    "reason":  final_reason,
                }
        return ready

    def poll_local_result(self):
        """
        Non-blocking lpop from local_results:queue; fetches task:{id}:local_result.
        Returns (task_id, result_dict) if available, else None.
        result_dict keys: task_type, qos_value, deadline_s, status, latency, energy, reason
        """
        task_id = self.r.lpop("local_results:queue")
        if not task_id:
            return None
        data = self.r.hgetall(f"task:{task_id}:local_result")
        if not data:
            return task_id, {}
        return task_id, {
            "task_type": data.get("task_type", "UNKNOWN"),
            "qos_value": float(data.get("qos_value", 0.0)),
            "deadline_s": float(data.get("deadline_s", 0.0)),
            "status":    data.get("status",  "FAILED"),
            "latency":   float(data.get("latency", 999.0)),
            "energy":    float(data.get("energy",  0.0)),
            "reason":    data.get("reason",  "UNKNOWN"),
        }

    def read_offload_mode(self) -> str:
        """
        Read sim:offload_mode written by the simulator at startup.
        Returns one of "heuristic", "allOffload", "allLocal", or "unknown".
        """
        val = self.r.get("sim:offload_mode")
        return val if val else "unknown"

    def close(self):
        self.r.close()
        print("[Redis-ENV] Connection closed")
