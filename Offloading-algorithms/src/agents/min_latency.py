"""
MinLatencyAgent — Greedy-Latency heuristic baseline.

Estimates task completion latency for each candidate node and selects the
minimum. Assumes nominal channel conditions (100 Mbps V2I, 50 Mbps V2V).

Latency model:
    RSU : tx_v2i + cpu_req_Mcycles / cpu_avail_MIPS
    SV  : tx_v2v + cpu_req_Mcycles / cpu_avail_MIPS

Reference:
  Mao et al., "Deep Reinforcement Learning for Online Computation Offloading
  in Wireless Powered MEC Networks", IEEE TMC, vol. 19, no. 11, pp. 2581–2595,
  Nov. 2020. (Min-delay heuristic baseline, Section V-B)

Note: Fixed channel assumptions (100/50 Mbps) match the baseline definition in
the above paper. This is intentional — a real heuristic without instantaneous
CSI would assume nominal channel conditions.

Supports both action-space layouts:
  Redis:  [RSU_0 .. RSU_{N-1}, SV_0 .. SV_{K-1}]  → rsus is a list
  Dummy:  [SV_0 .. SV_{K-1}, RSU, Local]           → rsus is a single object
"""

import numpy as np
from src.config import Config


def _first_valid(mask):
    """Return the first valid action index, or 0 as a last-resort fallback."""
    valid = np.where(mask == 1)[0]
    return int(valid[0]) if len(valid) > 0 else 0


class MinLatencyAgent:
    """
    Estimates task completion latency for each candidate node and selects the minimum.
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
                    proc = cpu_req / max(rsu.cpu_avail, 1e-6)
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
