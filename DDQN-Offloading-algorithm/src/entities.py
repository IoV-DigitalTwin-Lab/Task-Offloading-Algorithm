import math
from src.config import Config

class Task:
    def __init__(self, task_id, size, cpu_req, vehicle_id, deadline, qos, created_time):
        self.task_id = task_id
        self.size = size                
        self.cpu_req = cpu_req          
        self.origin_vehicle_id = vehicle_id
        self.deadline = deadline
        self.qos = qos 
        self.created_time = created_time

class Vehicle:
    def __init__(self, v_id, cpu_total, memory_total):
        self.v_id = v_id
        # Resources
        self.cpu_total = cpu_total
        self.cpu_avail = cpu_total
        self.memory_total = memory_total
        self.memory_avail = memory_total
        self.battery_total = Config.MAX_BATTERY
        self.battery_avail = Config.MAX_BATTERY
        self.current_tasks = 0
        
        # Kinematics
        self.speed = 0.0        # m/s
        self.pos_x = 0.0        # m
        self.pos_y = 0.0        # m
        self.heading = 0.0      # Degrees (0-360)
        self.acceleration = 0.0 # m/s^2
        self.connected_rsu_id = None
        
    def to_relative_feature_vector(self, rsu_x, rsu_y):
        """
        Returns normalized vector:
        [CPU, Mem, Battery, Speed, PosX, PosY, Heading, Accel, Tasks]
        And relative positions to RSU.
        """
        rel_x = self.pos_x - rsu_x
        rel_y = self.pos_y - rsu_y  #Returns coordinates RELATIVE to the RSU.
        return [
            self.cpu_avail / 5000.0,
            self.memory_avail / Config.MAX_MEMORY,
            self.battery_avail / Config.MAX_BATTERY,
            self.speed / Config.MAX_SPEED,
            rel_x / Config.RSU_RANGE,
            rel_y / Config.RSU_RANGE,
            self.heading / 360.0,              # Normalized Heading
            (self.acceleration + 2) / 4.0,     # Normalized Accel (-2 to +2 -> 0 to 1)
            self.current_tasks / 10.0
        ]

class RSU:
    def __init__(self, rsu_id, pos_x, pos_y, cpu_total, memory_total):
        self.rsu_id = rsu_id
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.cpu_total = cpu_total
        self.cpu_avail = cpu_total
        self.memory_total = memory_total
        self.memory_avail = memory_total
        self.queue_length = 0
        
    def to_feature_vector(self):
        return [
            self.cpu_avail / 10000.0,
            self.memory_avail / 10000.0,
            self.queue_length / 50.0
        ]