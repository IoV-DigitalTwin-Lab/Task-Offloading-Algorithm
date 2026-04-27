"""
LeastQueueAgent — Join-the-Shortest-Queue (JSQ) load-balancing baseline.

Picks the valid node with the shortest task queue. Works well as a near-optimal
static policy under homogeneous workloads.

References:
  Mitzenmacher, "The Power of Two Choices in Randomized Load Balancing",
  IEEE TPDS, vol. 12, no. 10, pp. 1094–1104, Oct. 2001.

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


class LeastQueueAgent:
    """Picks the valid node with the shortest task queue (JSQ policy)."""

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
