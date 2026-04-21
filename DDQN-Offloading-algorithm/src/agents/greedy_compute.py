"""
GreedyComputeAgent — picks the valid node with the highest available CPU (MIPS).

Standard "Max-Compute Greedy" heuristic used as a baseline in MEC/IoV task
offloading papers. Described as "selects the node with maximum available
computation capacity" — no single authoritative citation; common in the field.

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


class GreedyComputeAgent:
    """Picks the valid node with the highest available CPU (MIPS)."""

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
            # RSUs have ~10× more CPU; give a 10% bonus to bias towards offloading
            if rsu_idx < len(mask) and mask[rsu_idx] == 1 and (rsu.cpu_avail * 1.1) > best_cpu:
                action = rsu_idx

        return action
