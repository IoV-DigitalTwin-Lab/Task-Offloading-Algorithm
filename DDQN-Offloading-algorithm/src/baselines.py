from src.config import Config

class GreedyAgent:
    def select_action(self, candidates, rsu):
        """
        Greedy Strategy:
        1. Look at all Top-K candidates.
        2. Pick the one with the HIGHEST Available CPU.
        3. If RSU has more CPU than any vehicle, pick RSU.
        """
        best_cpu = -1
        action = Config.ACTION_DIM - 1 # Default to Drop/Fail
        
        # 1. Check Top-K Vehicle Candidates
        for i, vehicle in enumerate(candidates):
            if vehicle.cpu_avail > best_cpu:
                best_cpu = vehicle.cpu_avail
                action = i
                
        # 2. Check RSU (Action Index = MAX_NEIGHBORS)
        # We multiply by 1.2 bias because RSUs usually have faster links
        if (rsu.cpu_avail * 1.2) > best_cpu:
            action = Config.MAX_NEIGHBORS
            
        return action