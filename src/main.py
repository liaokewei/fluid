import taichi as ti
from gui import SimulationGUI
from liquid_simulator import WaterSurfaceSimulator
from scene import Boat, ObstacleManager
from rain import RainSimulator
import os

if __name__ == "__main__":
    ti.init(arch=ti.cuda, device_memory_fraction=0.8)

    water_res_x = 256
    water_res_z = 256
    
    water_sim = WaterSurfaceSimulator(water_res_x, water_res_z)
    rain_sim = RainSimulator(water_sim, probability=0.3, strength=-1.0, radius=2)
    
    relative_model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'model.obj')
    model_path = os.path.abspath(relative_model_path)
    
    # 您可以随时在这里调整船的初始状态
    boat_pos = [water_res_x * 0.5, 100.0, water_res_z * 0.9]
    boat_rot_degrees = [270, 60, 90]
    
    boat = Boat(
        obj_path=model_path,
        initial_pos=boat_pos,
        initial_rotation=boat_rot_degrees,
        water_bounds=(water_res_x, water_res_z)
    )
    
    obstacle_manager = ObstacleManager(max_obstacles=20, spawn_area=(water_res_x, water_res_z))

    gui = SimulationGUI(
        water_sim=water_sim,
        boat=boat,
        rain_sim=rain_sim,
        obstacle_manager=obstacle_manager,
        resolution=(1024, 768),
        title="Taichi Water Simulation"
    )
    gui.run()