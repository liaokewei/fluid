import taichi as ti
from renderer import WaterRenderer
from particle_system import ParticleSystem
from PIL import Image
import numpy as np

@ti.data_oriented
class SimulationGUI:
    def __init__(self, water_sim, boat, obstacle_manager, rain_sim, smoke_system, resolution, title):
        # 1. 保存所有模拟模块的实例
        self.water_sim = water_sim
        self.boat = boat
        self.obstacle_manager = obstacle_manager
        self.rain_sim = rain_sim
        self.smoke_system = smoke_system
        
        # 2. 初始化模拟参数和状态
        self.resolution = resolution
        self.dt = 1 / 60.0
        self.game_over = False
        self.is_raining = False
        self.is_windy = False

        # 3. 在GUI内部创建渲染器实例
        self.renderer = WaterRenderer(
            water_sim=self.water_sim,
            boat=self.boat,
            obstacle_manager=self.obstacle_manager,
            smoke_system=self.smoke_system
        )
        
        # 4. 初始化窗口、相机和背景图
        self.window = ti.ui.Window(title, resolution, vsync=True)
        self.canvas = self.window.get_canvas()
        self.camera = ti.ui.Camera()
        
        try:
            bg_image_path = '../model/OIP.jpg'
            bg_image = Image.open(bg_image_path).convert('RGB')
            bg_image = bg_image.resize(resolution)
            self.bg_image_np = (np.array(bg_image, dtype=np.float32) / 255.0)
        except FileNotFoundError:
            print(f"Warning: Background image not found at '{bg_image_path}'. Will use a solid color background.")
            self.bg_image_np = None

        # 5. 初始化各个模块
        self.water_sim.init_height_field()
        self.obstacle_manager.init_obstacles()
        self.smoke_system.init_particles()
        self.reset_camera()

    def reset_camera(self):
        center_x = self.water_sim.size_x * 0.5
        center_z = self.water_sim.size_z * 0.5
        self.camera.position(center_x, 80, center_z + 200)
        self.camera.lookat(center_x, 0.0, center_z)
        self.camera.up(0, 1, 0)
        self.camera.fov = 75

    def run(self):
        scene = self.window.get_scene()
        while self.window.running:
            # --- 1. 事件处理 ---
            for e in self.window.get_events(ti.ui.PRESS):
                if e.key == 'k':
                    if self.boat.is_sinking[None] == 0:
                        self.boat.is_sinking[None] = 1
            
                    

            # --- 2. 游戏与物理逻辑 ---
            if not self.game_over:
                move_dir = ti.Vector([0.0, 0.0])
                if self.window.is_pressed('a'): move_dir.x = -1.0
                if self.window.is_pressed('d'): move_dir.x = 1.0
                if self.window.is_pressed('w'): move_dir.y = -1.0
                if self.window.is_pressed('s'): move_dir.y = 1.0
                self.boat.control(move_dir, self.dt)
                
                self.boat.step(self.dt, self.water_sim.h)
                
                self.obstacle_manager.try_spawn(self.dt)
                self.obstacle_manager.update_obstacles(self.dt)
                
                self.water_sim.create_wakes_kernel(self.boat.position[None], self.boat.prev_position[None], self.obstacle_manager.obstacles)

                self.obstacle_manager.check_collisions(self.boat.position[None], self.boat.bounding_radius)
                if self.obstacle_manager.collision_flag[None] == 1:
                    print("Game Over!")
                    self.game_over = True
                    self.boat.is_sinking[None] = 1
                    self.obstacle_manager.collision_flag[None] = 0

            # --- 3. 天气与特殊状态模拟 ---
            if self.is_raining:
                self.rain_sim.step()
            
            if self.is_windy:
                self.water_sim.apply_wind()
                wind_dir = self.water_sim.wind_direction
                wind_force = wind_dir.normalized() * self.water_sim.wind_strength
                self.boat.velocity[None].x += wind_force.x * self.boat.wind_influence_factor * self.dt
                self.boat.velocity[None].z += wind_force.y * self.boat.wind_influence_factor * self.dt

            if self.boat.is_sinking[None] == 1:
                self.boat.sink(self.dt, self.water_sim.h)

            # --- 4. 核心模拟器步进与数据更新 ---
            self.water_sim.step()
            self.smoke_system.update(self.dt)
            if not self.game_over or self.boat.is_sinking[None] == 1:
                self.boat.update_world_space_data(self.water_sim.h)

            # --- 5. 渲染 ---
            if self.bg_image_np is not None:
                self.canvas.set_image(self.bg_image_np)
            else: 
                self.canvas.set_background_color((0.0, 0.0, 0.0))

            scene.set_camera(self.camera)
            scene.ambient_light((0.3, 0.3, 0.4))
            light_pos = self.camera.curr_position + ti.Vector([0, 20, 0])
            scene.point_light(pos=light_pos, color=(0.9, 0.9, 0.9))
            
            # --- 核心修正：使用正确的 UI 语法 ---
            gui = self.window.get_gui()
            gui.begin("Controls", 0.05, 0.05, 0.2, 0.25)
            gui.text("Weather Controls")
            if gui.button("Toggle Rain"):
                self.is_raining = not self.is_raining
            gui.text(f"  Raining: {self.is_raining}")
            if gui.button("Toggle Wind"):
                self.is_windy = not self.is_windy
            gui.text(f"  Windy: {self.is_windy}")
            gui.end()
            # ------------------------------------
            
            self.renderer.render(scene, self.game_over, self.camera.curr_position, light_pos, sinking_flag=self.boat.is_sinking[None])

            # --- 6. 显示到窗口 ---
            self.canvas.scene(scene)
            self.window.show()