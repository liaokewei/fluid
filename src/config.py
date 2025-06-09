# config.py

# 统一管理所有可调整的参数

# 游戏窗口设置
GUI_CONFIG = {
    "resolution": (1024, 768),
    "title": "Taichi Water Simulation"
}

# 水面模拟器设置
WATER_CONFIG = {
    "size_x": 256,
    "size_z": 256,
    "base_amplitude": 2.5,   # 整体浪高
    "base_frequency": 0.02,  # 整体波浪大小（频率越小，波浪越大越舒展）
    "persistence": 0.5,      # 细节的衰减度
    "time_scale": 0.3,       # 时间流逝速度
    "flow_speed": 0.8        # 水流速度
}

# 小船设置
BOAT_CONFIG = {
    # 使用您最终确定的参数
    "initial_pos": [WATER_CONFIG["size_x"] * 0.5, 100.0, WATER_CONFIG["size_z"] * 0.9],
    "initial_rotation": [270, 60, 90],
    "desired_scale": 20.0,
    "float_height": 40.0,
    "bounding_radius": 12.0,
    "move_acc": 800.0,
    "drag": 0.95,
    "buoyancy_spring": 250.0,
    "buoyancy_damping": 0.95
}

# 障碍物管理器设置
OBSTACLE_CONFIG = {
    "max_obstacles": 20,
    # 使用您最终确定的参数
    "group_spawn_interval": 5.0,
    "min_per_group": 4,
    "max_per_group": 6,
    "num_lanes": 8,
    "velocity": [0.0, 0.0, 45.0],
    "size_min": 8.0,
    "size_max": 12.0
}