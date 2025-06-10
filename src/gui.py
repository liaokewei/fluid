import taichi as ti
from renderer import WaterRenderer
from particle_system import ParticleSystem # 确保导入了 ParticleSystem

@ti.data_oriented
class SimulationGUI:
    def __init__(self, water_sim, boat, obstacle_manager, rain_sim, smoke_system, resolution, title):
        # --- 1. 保存所有模拟模块的实例 ---
        self.water_sim = water_sim
        self.boat = boat
        self.obstacle_manager = obstacle_manager
        self.rain_sim = rain_sim
        self.smoke_system = smoke_system
        
        # --- 2. 初始化模拟参数和状态 ---
        self.resolution = resolution
        self.dt = 1 / 60.0
        self.game_over = False
        self.is_raining = False
        self.is_windy = False

        # --- 3. 在GUI内部创建渲染器实例 ---
        self.renderer = WaterRenderer(
            water_sim=self.water_sim,
            boat=self.boat,
            obstacle_manager=self.obstacle_manager,
            smoke_system=self.smoke_system
        )
        
        # --- 4. 初始化窗口和相机 ---
        self.window = ti.ui.Window(title, resolution, vsync=True)
        self.canvas = self.window.get_canvas()
        self.camera = ti.ui.Camera()
        
        # --- 5. 初始化各个模块 ---
        self.water_sim.init_height_field()
        self.obstacle_manager.init_obstacles()
        self.smoke_system.init_particles()
        self.reset_camera()

    def reset_camera(self):
        center_x = self.water_sim.size_x * 0.5
        center_z = self.water_sim.size_z * 0.5
        self.camera.position(center_x, 80, center_z + 200) # 降低了相机高度以获得更好的视角
        self.camera.lookat(center_x, 0.0, center_z)
        self.camera.up(0, 1, 0)
        self.camera.fov = 75


    def run(self):
        scene = self.window.get_scene()
        while self.window.running:
            # 1. 事件处理
            if self.window.get_event(ti.ui.PRESS):
                if self.window.event.key == 'r':
                    self.is_raining = not self.is_raining
                if self.window.event.key == 'f':
                    self.is_windy = not self.is_windy

            # 2. 游戏与物理逻辑
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
                # self.create_wakes_kernel()
                self.water_sim.create_wakes_kernel(
                    self.boat.position[None], 
                    self.boat.prev_position[None], 
                    self.obstacle_manager.obstacles
                )
                self.obstacle_manager.check_collisions(self.boat.position[None], self.boat.bounding_radius)
                if self.obstacle_manager.collision_flag[None] == 1:
                    print("Game Over!")
                    self.game_over = True
                    self.smoke_system.spawn(self.boat.position[None])
                    self.obstacle_manager.collision_flag[None] = 0

            # 3. 天气效果模拟
            if self.is_raining:
                self.rain_sim.step()
            
            if self.is_windy:
                self.water_sim.apply_wind()
                wind_dir = self.water_sim.wind_direction
                wind_force = wind_dir.normalized() * self.water_sim.wind_strength
                self.boat.velocity[None].x += wind_force.x * self.boat.wind_influence_factor * self.dt
                self.boat.velocity[None].z += wind_force.y * self.boat.wind_influence_factor * self.dt

            # 4. 核心模拟器步进
            self.water_sim.step()
            self.smoke_system.update(self.dt)
            if not self.game_over:
                self.boat.update_world_space_data(self.water_sim.h)

            # 5. 渲染
            # print(f"Calculated Buoyant Force: {self.boat.debug_force_y[None]:.2f}")
            # print(f"Boat Pos: ({self.boat.debug_pos_x[None]:.2f}, {self.boat.debug_pos_z[None]:.2f}), Buoyant Force: {self.boat.debug_force_y[None]:.2f}")
        # ---------------------------------------------
            scene.set_camera(self.camera)
            scene.ambient_light((0.2, 0.2, 0.2))
            light_pos = self.camera.curr_position
            scene.point_light(pos=self.camera.curr_position, color=(0.8, 0.8, 0.8))
            self.renderer.render(scene, self.game_over, self.camera.curr_position, light_pos)

            # 6. 显示
            self.canvas.scene(scene)
            self.window.show()