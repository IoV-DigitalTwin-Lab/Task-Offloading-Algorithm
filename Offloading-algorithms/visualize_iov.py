import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import numpy as np
import random
from src.environment import IoVDummyEnv
from src.config import Config

class IoVVisualizer:
    def __init__(self):
        self.env = IoVDummyEnv()
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        
        # Setup Plot Boundaries
        self.ax.set_xlim(0, Config.MAP_WIDTH)
        self.ax.set_ylim(-Config.RSU_RANGE - 100, Config.RSU_RANGE + 100)
        self.ax.set_title("Vehicular Edge Computing Simulation (DeepRL Environment)")
        self.ax.set_xlabel("Highway Position (Meters)")
        self.ax.set_ylabel("Lateral Distance (Meters)")
        
        # Graphic Elements containers
        self.vehicle_scatter = None
        self.rsu_markers = []
        self.connection_line = None
        self.range_circles = []
        self.text_info = None
        
        # Initialize static elements (RSUs and Road)
        self._init_static_elements()

    def _init_static_elements(self):
        """Draws RSUs, Ranges, and Road borders once."""
        # Draw Road Borders
        self.ax.axhline(y=Config.RSU_RANGE, color='gray', linestyle='--', alpha=0.5)
        self.ax.axhline(y=-Config.RSU_RANGE, color='gray', linestyle='--', alpha=0.5)
        self.ax.axhline(y=0, color='black', linestyle='-', alpha=0.3) # Center line

        # Draw RSUs and their Coverage Zones
        for rsu in self.env.rsus:
            # RSU Marker (Blue Triangle)
            self.ax.plot(rsu.pos_x, rsu.pos_y, marker='^', color='blue', markersize=12, label='RSU' if rsu.rsu_id==0 else "")
            self.ax.text(rsu.pos_x, rsu.pos_y + 50, f"RSU {rsu.rsu_id}", ha='center', color='blue')
            
            # Coverage Circle
            circle = patches.Circle((rsu.pos_x, rsu.pos_y), Config.RSU_RANGE, 
                                    edgecolor='blue', facecolor='cyan', alpha=0.05)
            self.ax.add_patch(circle)

        # Initialize dynamic plot objects
        self.vehicle_scatter = self.ax.scatter([], [], c='gray', s=50, alpha=0.6)
        self.task_vehicle_scatter = self.ax.scatter([], [], c='red', marker='*', s=150, label='Task Source')
        self.target_line, = self.ax.plot([], [], 'g--', linewidth=2, alpha=0.8)
        self.text_info = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes, 
                                      bbox=dict(facecolor='white', alpha=0.8))

    def get_action_strategy(self):
        """
        Since we don't have the trained model file, we implement a 
        Heuristic Strategy for visualization purposes:
        1. Prefer Neighbor if battery > 50%
        2. Else, use RSU
        """
        # Logic matches environment.py action space
        # 0 to K-1: Neighbor vehicles
        # K: RSU
        
        if len(self.env.candidates) > 0:
            # Pick a random candidate for visualization variety
            return random.randint(0, len(self.env.candidates) - 1)
        else:
            # Default to RSU (Index = Max Neighbors)
            return Config.MAX_NEIGHBORS

    def update(self, frame):
        # 1. Step the Environment
        # reset() moves physics globally and picks a new task scenario
        state = self.env.reset() 
        
        # 2. Decide Action (Heuristic/Random since no model provided)
        action = self.get_action_strategy()
        
        # 3. Apply Action
        next_state, reward, done, info = self.env.step(action)

        # --- VISUALIZATION UPDATE ---
        
        # A. Update All Vehicles positions
        all_x = [v.pos_x for v in self.env.vehicles]
        all_y = [v.pos_y for v in self.env.vehicles]
        # Color code: Grey normally
        colors = ['gray'] * len(self.env.vehicles)
        sizes = [50] * len(self.env.vehicles)
        
        # B. Highlight Specific Actors
        task_v = self.env.task_origin_vehicle
        active_rsu = self.env.active_rsu
        
        target_x, target_y = active_rsu.pos_x, active_rsu.pos_y
        target_name = "RSU"

        # Determine who the target was based on action
        if action < len(self.env.candidates):
            target_v = self.env.candidates[action]
            target_x, target_y = target_v.pos_x, target_v.pos_y
            target_name = f"Veh {target_v.v_id}"
            
            # Highlight Candidate in Green
            # Find index in full list to update color
            for idx, v in enumerate(self.env.vehicles):
                if v.v_id == target_v.v_id:
                    colors[idx] = 'green'
                    sizes[idx] = 100

        # C. Draw Lines (Task -> RSU -> Target)
        # Note: In VEC, usually Task -> RSU (decision) -> Target
        # We draw a line from Source to Target to visualize the flow
        self.target_line.set_data([task_v.pos_x, target_x], [task_v.pos_y, target_y])

        # D. Update Scatter Plot
        self.vehicle_scatter.set_offsets(np.c_[all_x, all_y])
        self.vehicle_scatter.set_color(colors)
        self.vehicle_scatter.set_sizes(sizes)
        
        # Highlight Task Source separately (Star icon)
        self.task_vehicle_scatter.set_offsets(np.c_[task_v.pos_x, task_v.pos_y])

        # E. Update Info Text
        status_text = (
            f"Step: {frame}\n"
            f"Active RSU: {active_rsu.rsu_id}\n"
            f"Task Size: {self.env.current_task.size:.2f} MB\n"
            f"Decision: Offload to {target_name}\n"
            f"Latency: {info['latency']:.3f}s (Deadline: {self.env.current_task.deadline:.2f}s)\n"
            f"Reward: {reward:.2f}\n"
            f"Success: {'YES' if info['success'] else 'NO'}"
        )
        self.text_info.set_text(status_text)
        
        return self.vehicle_scatter, self.task_vehicle_scatter, self.target_line, self.text_info

    def animate(self):
        # Interval is in milliseconds (200ms = 5 FPS for slower, readable viz)
        anim = animation.FuncAnimation(self.fig, self.update, frames=200, interval=600, blit=False)
        plt.show()

if __name__ == "__main__":
    viz = IoVVisualizer()
    viz.animate()