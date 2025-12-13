import numpy as np
import random
import math
from src.config import Config

class RandomAgent:
    def select_action(self, mask):
        """Picks any valid action randomly."""
        valid_indices = np.where(mask == 1)[0]
        if len(valid_indices) > 0:
            return np.random.choice(valid_indices)
        return Config.ACTION_DIM - 1 # Drop

class GreedyComputeAgent:
    def select_action(self, candidates, rsu, mask):
        """Picks valid action with Highest CPU."""
        best_cpu = -1
        action = Config.ACTION_DIM - 1
        
        # Check Candidates
        for i, v in enumerate(candidates):
            if i < Config.MAX_NEIGHBORS and mask[i] == 1:
                if v.cpu_avail > best_cpu:
                    best_cpu = v.cpu_avail
                    action = i
        
        # Check RSU
        rsu_idx = Config.MAX_NEIGHBORS
        if mask[rsu_idx] == 1 and (rsu.cpu_avail * 1.1) > best_cpu:
            action = rsu_idx
            
        return action

class GreedyDistanceAgent:
    def select_action(self, candidates, rsu, mask):
        """
        Picks valid action that is physically CLOSEST.
        (Minimizes Transmission Time).
        """
        # Distance is implicitly handled by the Environment sorting candidates by distance
        # So Candidate[0] is always the closest vehicle.
        
        # 1. Try Closest Vehicle
        for i, v in enumerate(candidates):
            if mask[i] == 1:
                return i
                
        # 2. If no vehicle valid, try RSU
        if mask[Config.MAX_NEIGHBORS] == 1:
            return Config.MAX_NEIGHBORS
            
        return Config.ACTION_DIM - 1
    
class LocalAgent:
    def select_action(self, mask):
        """
        Always chooses Local Execution (Index = MAX_NEIGHBORS + 1).
        If Local is invalid (dead battery), it falls back to Random.
        """
        local_action = Config.MAX_NEIGHBORS + 1
        
        if mask[local_action] == 1:
            return local_action
            
        # Fallback if battery is dead
        valid_indices = np.where(mask == 1)[0]
        if len(valid_indices) > 0:
            return np.random.choice(valid_indices)
        return local_action