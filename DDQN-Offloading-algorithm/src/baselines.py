import numpy as np
from src.config import Config


def _first_valid(mask):
    """Return the first valid action index, or 0 as a last-resort fallback."""
    valid = np.where(mask == 1)[0]
    return int(valid[0]) if len(valid) > 0 else 0


class RandomAgent:
    """Selects a uniformly random valid action."""
    def select_action(self, mask):
        valid = np.where(mask == 1)[0]
        return int(np.random.choice(valid)) if len(valid) > 0 else 0


class GreedyComputeAgent:
    """
    Picks the valid node with the highest available CPU (MIPS).

    Supports both action-space layouts:
      Redis:  [RSU_0 .. RSU_{N-1}, SV_0 .. SV_{K-1}]  → rsus is a list
      Dummy:  [SV_0 .. SV_{K-1}, RSU, Local]           → rsus is a single object
    """
    def select_action(self, candidates, rsus, mask):
        best_cpu = -1.0
        action = _first_valid(mask)

        if isinstance(rsus, list):
            # Redis layout
            for i, rsu in enumerate(rsus):
                if mask[i] == 1 and rsu.cpu_avail > best_cpu:
                    best_cpu, action = rsu.cpu_avail, i
            for j, v in enumerate(candidates):
                idx = len(rsus) + j
                if idx < len(mask) and mask[idx] == 1 and v.cpu_avail > best_cpu:
                    best_cpu, action = v.cpu_avail, idx
        else:
            # Dummy layout
            rsu = rsus
            for i, v in enumerate(candidates):
                if i < Config.MAX_NEIGHBORS and mask[i] == 1 and v.cpu_avail > best_cpu:
                    best_cpu, action = v.cpu_avail, i
            rsu_idx = Config.MAX_NEIGHBORS
            # RSUs have ~10× more CPU; give a 10% bonus to bias towards offload
            if rsu_idx < len(mask) and mask[rsu_idx] == 1 and (rsu.cpu_avail * 1.1) > best_cpu:
                action = rsu_idx

        return action


class MinLatencyAgent:
    """
    Estimates task completion latency for each candidate node and selects the minimum.

    Latency model (simplified):
        RSU : tx_v2i + cpu_req_Mcycles / cpu_avail_MIPS
        SV  : tx_v2v + cpu_req_Mcycles / cpu_avail_MIPS   (V2V relay adds overhead)

    Inspired by: Mao et al., "Deep Reinforcement Learning for Online Computation
    Offloading in Wireless Powered MEC Networks", IEEE TMC 2020.
    """
    _V2I_BW_MBPS = 100.0   # Vehicle-to-Infrastructure channel (Mbps)
    _V2V_BW_MBPS = 50.0    # Vehicle-to-Vehicle relay channel  (Mbps)

    def select_action(self, candidates, rsus, mask, task_info=None):
        """
        task_info: dict with keys 'cpu_req_mcycles' and 'mem_footprint_mb'.
                   Falls back to conservative defaults if None.
        """
        cpu_req = task_info['cpu_req_mcycles']  if task_info else 100.0  # Mcycles
        mem_mb  = task_info['mem_footprint_mb'] if task_info else 1.0    # MB

        # Transmission times (seconds)
        tx_v2i = (mem_mb * 8.0) / self._V2I_BW_MBPS
        tx_v2v = (mem_mb * 8.0) / self._V2V_BW_MBPS

        best_lat = float('inf')
        action = _first_valid(mask)

        if isinstance(rsus, list):
            # Redis layout: RSUs at 0..N-1, SVs at N..N+K-1
            for i, rsu in enumerate(rsus):
                if mask[i] == 1:
                    proc = cpu_req / max(rsu.cpu_avail, 1e-6)  # Mcycles/MIPS = seconds
                    lat  = tx_v2i + proc
                    if lat < best_lat:
                        best_lat, action = lat, i
            for j, v in enumerate(candidates):
                idx = len(rsus) + j
                if idx < len(mask) and mask[idx] == 1:
                    proc = cpu_req / max(v.cpu_avail, 1e-6)
                    lat  = tx_v2v + proc
                    if lat < best_lat:
                        best_lat, action = lat, idx
        else:
            # Dummy layout: SVs at 0..K-1, RSU at K
            rsu = rsus
            rsu_idx = Config.MAX_NEIGHBORS
            if rsu_idx < len(mask) and mask[rsu_idx] == 1:
                proc = cpu_req / max(rsu.cpu_avail, 1e-6)
                best_lat, action = tx_v2i + proc, rsu_idx
            for i, v in enumerate(candidates):
                if i < Config.MAX_NEIGHBORS and mask[i] == 1:
                    proc = cpu_req / max(v.cpu_avail, 1e-6)
                    lat  = tx_v2v + proc
                    if lat < best_lat:
                        best_lat, action = lat, i

        return action


class LeastQueueAgent:
    """
    Picks the valid node with the shortest task queue (load-balancing baseline).

    Inspired by: Join-the-Shortest-Queue (JSQ) policy from queuing theory.
    Works well as a near-optimal static policy under homogeneous workloads.
    """
    def select_action(self, candidates, rsus, mask):
        best_q = float('inf')
        action = _first_valid(mask)

        if isinstance(rsus, list):
            # Redis layout
            for i, rsu in enumerate(rsus):
                if mask[i] == 1 and rsu.queue_length < best_q:
                    best_q, action = rsu.queue_length, i
            for j, v in enumerate(candidates):
                idx = len(rsus) + j
                if idx < len(mask) and mask[idx] == 1 and v.queue_length < best_q:
                    best_q, action = v.queue_length, idx
        else:
            # Dummy layout — Vehicle uses 'current_tasks', RSU uses 'queue_length'
            rsu = rsus
            rsu_idx = Config.MAX_NEIGHBORS
            if rsu_idx < len(mask) and mask[rsu_idx] == 1:
                best_q = getattr(rsu, 'queue_length', 0)
                action = rsu_idx
            for i, v in enumerate(candidates):
                if i < Config.MAX_NEIGHBORS and mask[i] == 1:
                    q = getattr(v, 'queue_length', getattr(v, 'current_tasks', 0))
                    if q < best_q:
                        best_q, action = q, i

        return action


class GreedyDistanceAgent:
    """
    Picks the physically closest valid node (candidates are pre-sorted by distance).
    Kept for backward compatibility with the dummy env.
    Prefer MinLatencyAgent for better performance.
    """
    def select_action(self, candidates, rsus, mask):
        if isinstance(rsus, list):
            # Redis layout: try SVs first (already sorted by distance), then RSUs
            for j in range(len(candidates)):
                idx = len(rsus) + j
                if idx < len(mask) and mask[idx] == 1:
                    return idx
            for i in range(len(rsus)):
                if mask[i] == 1:
                    return i
        else:
            # Dummy layout
            for i in range(len(candidates)):
                if i < len(mask) and mask[i] == 1:
                    return i
            rsu_idx = Config.MAX_NEIGHBORS
            if rsu_idx < len(mask) and mask[rsu_idx] == 1:
                return rsu_idx

        return _first_valid(mask)


class LocalAgent:
    """Always chooses local execution (dummy env only; no local slot in redis env)."""
    def select_action(self, mask):
        local_action = Config.MAX_NEIGHBORS + 1
        if local_action < len(mask) and mask[local_action] == 1:
            return local_action
        valid = np.where(mask == 1)[0]
        return int(np.random.choice(valid)) if len(valid) > 0 else local_action
