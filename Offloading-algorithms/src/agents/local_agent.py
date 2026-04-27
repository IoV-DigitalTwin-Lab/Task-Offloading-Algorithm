"""
LocalAgent — All-Local Execution (ALE) baseline.

In the dummy environment: always selects the local execution action slot.
In the Redis environment: passive agent that skips offload decisions entirely
  and only collects local_results:queue entries for metric tracking.

Reference (All-Local baseline):
  Mao et al., "Stochastic Joint Radio and Computational Resource Management
  for Multi-User Mobile-Edge Computing Systems",
  IEEE JSAC, vol. 35, no. 6, pp. 1451–1465, Jun. 2017.
"""

import numpy as np
from src.config import Config


class LocalAgent:
    """
    All-Local Execution baseline agent.

    Dummy env: select_action(mask) always returns the local execution slot.
    Redis env: poll_and_collect(env) polls local_results:queue passively.
               The main loop skips offload decisions for this agent entirely.
    """

    name = "local"

    # ── Dummy environment interface ───────────────────────────────────────────

    def select_action(self, mask):
        """Always chooses local execution (dummy env only)."""
        local_action = Config.MAX_NEIGHBORS + 1
        if local_action < len(mask) and mask[local_action] == 1:
            return local_action
        valid = np.where(mask == 1)[0]
        return int(np.random.choice(valid)) if len(valid) > 0 else local_action

    # ── Redis environment interface ───────────────────────────────────────────

    def poll_and_collect(self, env):
        """
        Non-blocking poll of local_results:queue (Redis env only).

        Returns:
            (task_id, result_dict) if a result is available, else None.
            result_dict keys: task_type, qos_value, deadline_s, status,
                              latency, energy, reason
        """
        return env.poll_local_result()
