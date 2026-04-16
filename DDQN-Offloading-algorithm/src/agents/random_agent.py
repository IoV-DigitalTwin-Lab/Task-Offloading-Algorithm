"""
RandomAgent — selects a uniformly random valid action.

Used as a lower-bound reference baseline in comparisons.
No citation required; standard practice in MEC/IoV DRL literature.
"""

import numpy as np


class RandomAgent:
    """Selects a uniformly random valid action."""

    def select_action(self, mask):
        valid = np.where(mask == 1)[0]
        return int(np.random.choice(valid)) if len(valid) > 0 else 0
