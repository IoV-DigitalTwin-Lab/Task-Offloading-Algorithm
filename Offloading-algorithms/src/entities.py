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
        
        # Utilization metrics
        self.cpu_utilization = 0.0
        self.mem_utilization = 0.0
        self.processing_count = 0
        
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
        [CPU, Mem, Speed, PosX, PosY, Heading, Tasks, CPU_Util, Mem_Util, Processing]
        And relative positions to RSU.
        """
        rel_x = self.pos_x - rsu_x
        rel_y = self.pos_y - rsu_y  #Returns coordinates RELATIVE to the RSU.
        return [
            self.cpu_avail / 5000.0,
            self.memory_avail / Config.MAX_MEMORY,
            self.speed / Config.MAX_SPEED,
            rel_x / Config.RSU_RANGE,
            rel_y / Config.RSU_RANGE,
            self.heading / 360.0,              # Normalized Heading
            self.current_tasks / 10.0,
            self.cpu_utilization,              # Already 0.0-1.0 from DB
            self.mem_utilization,              # Already 0.0-1.0 from DB
            self.processing_count / 10.0       # Normalize processing count
        ]

class RSU:
    def __init__(self, rsu_id, pos_x, pos_y, cpu_total, memory_total, bandwidth=20.0):
        self.rsu_id = rsu_id
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.cpu_total = cpu_total
        self.cpu_avail = cpu_total
        self.memory_total = memory_total
        self.memory_avail = memory_total
        self.bandwidth = bandwidth
        self.queue_length = 0
        
        # Utilization and processing metrics
        self.cpu_utilization = 0.0
        self.memory_utilization = 0.0
        self.processing_count = 0
        self.max_concurrent_tasks = 10
        
    def to_feature_vector(self):
        return [
            self.cpu_avail / 10000.0,
            self.memory_avail / 10000.0,
            self.bandwidth / 100.0, # Normalize bandwidth
            self.queue_length / 50.0,
            self.cpu_utilization,    # Already 0.0-1.0 from DB
            self.memory_utilization, # Already 0.0-1.0 from DB
            self.processing_count / 20.0,
            self.max_concurrent_tasks / 20.0
        ]