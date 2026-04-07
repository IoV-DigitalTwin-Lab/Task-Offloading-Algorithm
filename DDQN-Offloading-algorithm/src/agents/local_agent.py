"""
LocalAgent: passive agent that only collects local execution results.

No offloading decisions are made. It polls local_results:queue and
accumulates per-task-type metrics for all-local or heuristic runs.
"""


class LocalAgent:
    """Passive agent: no decisions made. Collects local execution results only."""

    name = "local"

    def poll_and_collect(self, env):
        """
        Non-blocking poll of local_results:queue.

        Returns:
            (task_id, result_dict) if a result is available, else None.
            result_dict keys: task_type, qos_value, deadline_s, status,
                              latency, energy, reason
        """
        return env.poll_local_result()
