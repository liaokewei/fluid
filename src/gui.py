import taichi as ti
import math
import random
from scene import Boat, ObstacleManager
from particle_system import ParticleSystem 

@ti.data_oriented
class SimulationGUI:
    # ... (__init__ 和其他函数不变) ...
    def __init__(self, water_sim, boat, obstacle_manager, resolution, title):
        self.water_sim = water_sim
        self.boat = boat
        self.obstacle_manager = obstacle_manager
        self.smoke_system = ParticleSystem(max_particles=500)
        self.resolution = resolution
        self.dt = 1 / 60.0
        self.game_over = False
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
        self.color_contrast = 1.5
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
            self.water_vertices[idx] = ti.Vector([i, self.water_sim.h[i, j], j])
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
        scene.set_camera(self.camera)
        scene.point_light(pos=(self.water_sim.size_x, self.water_sim.size_z, self.water_sim.size_x), color=(1, 1, 1))
        scene.ambient_light((0.5, 0.5, 0.5))
        scene.mesh(self.water_vertices, indices=self.water_indices, per_vertex_color=self.water_vertex_colors, two_sided=True)
        if not self.game_over:
            scene.mesh(vertices=self.boat.world_vertices, indices=self.boat.indices, normals=self.boat.world_normals, color=self.boat_color)
            scene.particles(self.obstacle_manager.obstacles.position, radius=10.0, color=self.obstacle_color)
        
        # --- 核心修正：减小粒子半径 ---
        scene.particles(self.smoke_system.particles.position,
                        radius=0.3, # 原来是 0.8
                        color=self.smoke_color)
        # ---------------------------

    def run(self):
        scene = self.window.get_scene()
        while self.window.running:
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
                self.obstacle_manager.check_collisions(self.boat.position[None], self.boat.bounding_radius)
                if self.obstacle_manager.collision_flag[None] == 1:
                    print("Game Over!")
                    self.game_over = True
                    self.smoke_system.spawn(self.boat.position[None])
                    self.obstacle_manager.collision_flag[None] = 0
            self.water_sim.step()
            self.smoke_system.update(self.dt)
            if not self.game_over:
                self.boat.update_world_space_data()
            self.update_water_vertices()
            self.update_water_colors()
            self.render(scene)
            self.canvas.scene(scene)
            self.window.show()