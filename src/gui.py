import taichi as ti
import math
import random
from scene import Boat, ObstacleManager
from particle_system import ParticleSystem 

from PIL import Image
import numpy as np

@ti.data_oriented
class SimulationGUI:
    # ... (__init__ 和其他函数不变) ...
    def __init__(self, water_sim, boat, obstacle_manager, rain_sim, resolution, title):

        self.bg_image_path = '../model/OIP.jpg'  # 替换为你的背景图片路径
        self.bg_image = Image.open(self.bg_image_path).convert('RGB')
        self.bg_image = self.bg_image.resize(resolution)  # 调整大小以匹配窗口尺寸
        self.bg_image_np = np.array(self.bg_image)

        self.water_sim = water_sim
        self.boat = boat
        self.obstacle_manager = obstacle_manager
        self.rain_sim = rain_sim
        self.smoke_system = ParticleSystem(max_particles=500)
        self.resolution = resolution
        self.dt = 1 / 60.0
        self.game_over = False

        self.is_raining = False
        self.is_windy = False

        self.window = ti.ui.Window(title, resolution, vsync=True)
        self.canvas = self.window.get_canvas()
        self.camera = ti.ui.Camera()
        num_vertices = self.water_sim.size_x * self.water_sim.size_z
        self.water_vertices = ti.Vector.field(3, dtype=ti.f32, shape=num_vertices)
        self.water_vertex_colors = ti.Vector.field(3, dtype=ti.f32, shape=num_vertices)
        num_triangles = (self.water_sim.size_x - 1) * (self.water_sim.size_z - 1) * 2
        self.water_indices = ti.field(dtype=ti.i32, shape=num_triangles * 3)
        self.crest_color = ti.Vector([0.6, 0.8, 1.0])
        self.trough_color = ti.Vector([0.2, 0.45, 0.75])
        self.color_contrast = 2.0

        self.visual_height_scale =15.0 #TODO

        self.boat_color = (0.45, 0.33, 0.22)
        self.obstacle_color = (0.9, 0.9, 0.95)
        self.smoke_color = (0.5, 0.5, 0.5)
        self.init_water_mesh()
        self.water_sim.init_height_field()
        self.obstacle_manager.init_obstacles()
        self.smoke_system.init_particles()
        self.reset_camera()
    def init_water_mesh(self):
        for i, j in ti.ndrange(self.water_sim.size_x - 1, self.water_sim.size_z - 1):
            quad_id = (i * (self.water_sim.size_z - 1)) + j
            v_idx_00 = i * self.water_sim.size_z + j
            v_idx_10 = (i + 1) * self.water_sim.size_z + j
            v_idx_01 = i * self.water_sim.size_z + (j + 1)
            v_idx_11 = (i + 1) * self.water_sim.size_z + (j + 1)
            self.water_indices[quad_id * 6 + 0] = v_idx_00
            self.water_indices[quad_id * 6 + 1] = v_idx_10
            self.water_indices[quad_id * 6 + 2] = v_idx_01
            self.water_indices[quad_id * 6 + 3] = v_idx_10
            self.water_indices[quad_id * 6 + 4] = v_idx_11
            self.water_indices[quad_id * 6 + 5] = v_idx_01
    def reset_camera(self):
        center_x = self.water_sim.size_x * 0.5
        center_z = self.water_sim.size_z * 0.5
        self.camera.position(center_x, 150, center_z + 120)
        self.camera.lookat(center_x, 0.0, center_z)
        self.camera.up(0, 1, 0)
        self.camera.fov = 60
    @ti.kernel
    def update_water_vertices(self):
        for i, j in self.water_sim.h:
            idx = i * self.water_sim.size_z + j
            # self.water_vertices[idx] = ti.Vector([i, self.water_sim.h[i, j], j])
            visual_height = self.water_sim.h[i, j] * self.visual_height_scale
            self.water_vertices[idx] = ti.Vector([i, visual_height, j])
    @ti.kernel
    def update_water_colors(self):
        for i, j in self.water_sim.h:
            idx = i * self.water_sim.size_z + j
            height = self.water_sim.h[i, j]
            normalized_h = height * self.color_contrast * 0.5 + 0.5
            t = ti.max(0.0, ti.min(1.0, normalized_h))
            final_color = self.crest_color * t + self.trough_color * (1.0 - t)
            self.water_vertex_colors[idx] = final_color

    def render(self, scene):
        self.canvas.set_image(self.bg_image_np)
        scene.set_camera(self.camera)
        scene.point_light(pos=(self.water_sim.size_x, self.water_sim.size_z, self.water_sim.size_x), color=(1, 1, 1))
        scene.ambient_light((0.5, 0.5, 0.5))
        scene.mesh(self.water_vertices, indices=self.water_indices, per_vertex_color=self.water_vertex_colors, two_sided=True)
        if not self.game_over or self.boat.is_sinking[None] == 1:
            # scene.mesh(vertices=self.boat.world_vertices, indices=self.boat.indices, normals=self.boat.world_normals, color=self.boat_color)
            scene.mesh(
                vertices=self.boat.world_vertices,
                indices=self.boat.indices,
                per_vertex_color=self.boat.vertex_color,
                # 其他参数比如 shading="flat" 等等
            )
            scene.particles(self.obstacle_manager.obstacles.position, radius=6.0, color=self.obstacle_color)
        
        # --- 核心修正：减小粒子半径 ---
        scene.particles(self.smoke_system.particles.position,
                        radius=0.3, # 原来是 0.8
                        color=self.smoke_color)
        # ---------------------------

    # ... 位于 SimulationGUI 类内 ...

    @ti.kernel
    def create_wakes_kernel(self):
        # 1. 为小船创建更真实的尾迹
        boat_pos = self.boat.position[None]
        boat_prev_pos = self.boat.prev_position[None]

        # 沿上一帧到当前帧的路径插值，模拟船体划过水面
        # 尾迹强度在船的后方最强，并向前逐渐减弱
        for i in range(5):
            t = i / 5.0  # t 从 0.0 变化到 0.8
            # t=0时在prev_pos, t=1时在pos。我们反过来，让t=0在后方
            pos = boat_pos * t + boat_prev_pos * (1.0 - t)

            bx, bz = ti.cast(pos.x, ti.i32), ti.cast(pos.z, ti.i32)
            radius = 4
            # 强度随着 t 增大（靠近船当前位置）而减弱
            strength = -0.08 * (1.0 - t) # 起始强度稍大，然后衰减

            for off_i, off_j in ti.ndrange((-radius, radius), (-radius, radius)):
                px, pz = bx + off_i, bz + off_j
                if 1 <= px < self.water_sim.size_x - 1 and 1 <= pz < self.water_sim.size_z - 1:
                    dist_sq = off_i * off_i + off_j * off_j
                    if dist_sq < radius**2:
                        falloff = ti.cos(0.5 * 3.14159265 * ti.sqrt(dist_sq) / radius)
                        ti.atomic_add(self.water_sim.v[px, pz], strength * falloff)

        # 2. 障碍物尾迹 (保持不变)
        for i in self.obstacle_manager.obstacles:
            if self.obstacle_manager.obstacles[i].active == 1:
                obs_pos = self.obstacle_manager.obstacles[i].position
                ox, oz = ti.cast(obs_pos.x, ti.i32), ti.cast(obs_pos.z, ti.i32)
                
                obs_radius = 3 
                obs_strength = -0.05

                for i_o, j_o in ti.ndrange((-obs_radius, obs_radius), (-obs_radius, obs_radius)):
                    px_o, pz_o = ox + i_o, oz + j_o
                    if 1 <= px_o < self.water_sim.size_x - 1 and 1 <= pz_o < self.water_sim.size_z - 1:
                        dist_sq_o = i_o*i_o + j_o*j_o
                        if dist_sq_o < obs_radius**2:
                            falloff_o = ti.cos(0.5 * 3.14159265 * ti.sqrt(dist_sq_o) / obs_radius)
                            ti.atomic_add(self.water_sim.v[px_o, pz_o], obs_strength * falloff_o)


    

    def run(self):
        scene = self.window.get_scene()
        while self.window.running:
            if self.window.get_event(ti.ui.PRESS):
                if self.window.event.key == 'r': # 按 'r' 键
                    self.is_raining = not self.is_raining # 切换下雨状态
                if self.window.event.key == 'f':
                    self.is_windy = not self.is_windy

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

                self.create_wakes_kernel()

                self.obstacle_manager.check_collisions(self.boat.position[None], self.boat.bounding_radius)
                if self.obstacle_manager.collision_flag[None] == 1:
                    print("Game Over!")
                    self.game_over = True
                    # self.smoke_system.spawn(self.boat.position[None])
                    self.boat.is_sinking[None] = 1
                    self.obstacle_manager.collision_flag[None] = 0

            if self.is_raining:
                self.rain_sim.step()

            if self.is_windy:
                # 1. 风吹拂湖面
                self.water_sim.apply_wind()

                # --- 2. 新增：风推动小船 ---
                wind_dir = self.water_sim.wind_direction
                wind_force = wind_dir.normalized() * self.water_sim.wind_strength
                
                # 将风力施加到小船的速度上
                self.boat.velocity[None].x += wind_force.x * self.boat.wind_influence_factor * self.dt
                self.boat.velocity[None].z += wind_force.y * self.boat.wind_influence_factor * self.dt
                # -------------------------

            self.water_sim.step()
            # self.smoke_system.update(self.dt)
            if self.boat.is_sinking[None] == 1:
                self.boat.sink(self.dt, self.water_sim.h)
            if not self.game_over or self.boat.is_sinking[None] == 1:
                self.boat.update_world_space_data()
            self.update_water_vertices()
            self.update_water_colors()
            self.render(scene)
            self.canvas.scene(scene)
            self.window.show()