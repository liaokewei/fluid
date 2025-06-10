import taichi as ti
import numpy as np
import trimesh
import os
import math
import random

from PIL import Image

@ti.data_oriented
class Boat:
    def __init__(self, obj_path: str, initial_pos, initial_rotation, water_bounds):
        self.position = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.velocity = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.rotation = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.prev_position = ti.Vector.field(3, dtype=ti.f32, shape=())
        
        self.position.from_numpy(np.array(initial_pos, dtype=np.float32))
        self.velocity[None] = [0.0, 0.0, 0.0]
        rot_radians = [math.radians(angle) for angle in initial_rotation]
        self.rotation.from_numpy(np.array(rot_radians, dtype=np.float32))

        # --- 物理与视觉参数 ---
        self.move_acc = 800.0
        self.drag = 0.95
        self.bounds = water_bounds
        self.float_height = 10.0
        self.visual_height_scale = 15.0
        self.buoyancy_spring = 400.0 # 使用一个经过验证的稳定值
        self.buoyancy_damping = 0.9
        # self.buoyancy_damping = 0.95
        self.bounding_radius = 12.0
        self.wind_influence_factor = 40.0

        self.debug_values = ti.Vector.field(4, dtype=ti.f32, shape=())
        
        # --- 浮力点定义 ---
        self.buoyancy_points = ti.Vector.field(3, dtype=ti.f32, shape=4)
        buoyancy_points_np = np.array([
            [0, 0, -15], [0, 0, 15], [-8, 0, 0], [8, 0, 0]
        ], dtype=np.float32)
        self.buoyancy_points.from_numpy(buoyancy_points_np)
        
        # --- 修正：将新增的“沉没”属性定义在 __init__ 中 ---
        self.is_sinking = ti.field(dtype=ti.i32, shape=())
        self.sink_velocity = ti.field(dtype=ti.f32, shape=())
        self.max_sink_depth = ti.field(dtype=ti.f32, shape=())
        self.is_sinking[None] = 0
        self.sink_velocity[None] = 0.0
        self.max_sink_depth[None] = 8.0
        # -----------------------------------------------

        self.local_vertices = None
        self.world_vertices = None
        self.local_normals = None
        self.world_normals = None
        self.uvs = None
        self.indices = None
        self.vertex_color = None
        
        self.load_model_from_obj(obj_path)
        self.initial_world_space_data() # 现在可以安全调用

    # --- 核心修正：将此函数变为 @ti.kernel ---
    @ti.kernel
    def initial_world_space_data(self):
        rot_matrix = create_rotation_matrix(self.rotation[None])
        for i in self.local_vertices:
            rotated_vertex = rot_matrix @ self.local_vertices[i]
            self.world_vertices[i] = self.position[None] + rotated_vertex
            rotated_normal = rot_matrix @ self.local_normals[i]
            self.world_normals[i] = rotated_normal.normalized()
    # ---------------------------------------

    def load_model_from_obj(self, filepath):
        mesh = trimesh.load_mesh(filepath, process=True)
        rotation_angle_rad = math.pi / 2.0
        rotation_matrix = trimesh.transformations.rotation_matrix(angle=rotation_angle_rad, direction=[1, 0, 0])
        mesh.apply_transform(rotation_matrix)
        center = mesh.vertices.mean(axis=0)
        mesh.vertices -= center
        max_extent = (mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)).max()
        mesh.vertices /= max_extent
        desired_scale = 20.0
        mesh.vertices *= desired_scale
        vertices_np = np.array(mesh.vertices, dtype=np.float32)
        indices_np = np.array(mesh.faces.flatten(), dtype=np.int32)
        normals_np = np.array(mesh.vertex_normals, dtype=np.float32)
        if mesh.visual.uv is not None and len(mesh.visual.uv) > 0:
            uvs_np = np.array(mesh.visual.uv, dtype=np.float32)
        else:
            uvs_np = np.zeros((len(vertices_np), 2), dtype=np.float32)
        num_vertices, num_indices = len(vertices_np), len(indices_np)
        self.local_vertices = ti.Vector.field(3, dtype=ti.f32, shape=num_vertices)
        self.world_vertices = ti.Vector.field(3, dtype=ti.f32, shape=num_vertices)
        self.local_normals = ti.Vector.field(3, dtype=ti.f32, shape=num_vertices)
        self.world_normals = ti.Vector.field(3, dtype=ti.f32, shape=num_vertices)
        self.uvs = ti.Vector.field(2, dtype=ti.f32, shape=num_vertices)
        self.indices = ti.field(dtype=ti.i32, shape=num_indices)
        self.local_vertices.from_numpy(vertices_np)
        self.indices.from_numpy(indices_np)
        self.local_normals.from_numpy(normals_np)
        self.uvs.from_numpy(uvs_np)

        # 贴图相关逻辑保持不变
        try:
            texture_image = Image.open("../model/PIC_01.png").convert("RGB")
            texture_data = np.array(texture_image, dtype=np.float32) / 255.0
            tex_h, tex_w = texture_data.shape[:2]

            uvs_np = self.uvs.to_numpy() 
            vertex_color_np = np.zeros((uvs_np.shape[0], 3), dtype=np.float32)
            for i, (u, v) in enumerate(uvs_np):
                u_ = np.clip(u, 0, 1)
                v_ = np.clip(v, 0, 1)
                x = int(u_ * (tex_w - 1))
                y = int((1 - v_) * (tex_h - 1))
                vertex_color_np[i] = texture_data[y, x, :3]
            self.vertex_color = ti.Vector.field(3, dtype=ti.f32, shape=uvs_np.shape[0])
            self.vertex_color.from_numpy(vertex_color_np)
        except FileNotFoundError:
            print("Warning: Texture file not found. Using default colors.")
            self.vertex_color = None


    @ti.kernel
    def update_world_space_data(self, water_h: ti.template()):
        base_rot_matrix = create_rotation_matrix(self.rotation[None])
        
        # --- 1. 计算俯仰角 (Pitch) ---
        p_front_world = self.position[None] + base_rot_matrix @ self.buoyancy_points[0]
        p_stern_world = self.position[None] + base_rot_matrix @ self.buoyancy_points[1]
        h_front = water_h[ti.cast(p_front_world.x, ti.i32), ti.cast(p_front_world.z, ti.i32)] * self.visual_height_scale
        h_stern = water_h[ti.cast(p_stern_world.x, ti.i32), ti.cast(p_stern_world.z, ti.i32)] * self.visual_height_scale
        pitch_angle = (h_front - h_stern) * 0.03

        # --- 2. 计算侧倾角 (Roll) ---
        p_port_world = self.position[None] + base_rot_matrix @ self.buoyancy_points[2]
        p_starboard_world = self.position[None] + base_rot_matrix @ self.buoyancy_points[3]
        h_port = water_h[ti.cast(p_port_world.x, ti.i32), ti.cast(p_port_world.z, ti.i32)] * self.visual_height_scale
        h_starboard = water_h[ti.cast(p_starboard_world.x, ti.i32), ti.cast(p_starboard_world.z, ti.i32)] * self.visual_height_scale
        roll_angle = (h_port - h_starboard) * 0.03
        
        # --- 3. 合成最终的旋转矩阵并更新顶点 ---
        tilt_rotation = ti.Vector([pitch_angle, 0.0, roll_angle])
        tilt_rot_matrix = create_rotation_matrix(tilt_rotation)
        final_rot_matrix = base_rot_matrix @ tilt_rot_matrix
        
        for i in self.local_vertices:
            rotated_vertex = final_rot_matrix @ self.local_vertices[i]
            rotated_normal = final_rot_matrix @ self.local_normals[i]
            self.world_vertices[i] = self.position[None] + rotated_vertex
            self.world_normals[i] = rotated_normal.normalized()

    @ti.kernel
    def step(self, dt: ti.f32, water_h: ti.template()):
        self.prev_position[None] = self.position[None]

        total_buoyancy_force = 0.0
        rot_matrix = create_rotation_matrix(self.rotation[None])
        
        for i in ti.static(range(4)):
            world_p = self.position[None] + rot_matrix @ self.buoyancy_points[i]
            grid_x = ti.max(0, ti.min(self.bounds[0] - 1, ti.cast(world_p.x, ti.i32)))
            grid_z = ti.max(0, ti.min(self.bounds[1] - 1, ti.cast(world_p.z, ti.i32)))
            water_level = water_h[grid_x, grid_z] * self.visual_height_scale
            displacement = (water_level + self.float_height) - world_p.y
            total_buoyancy_force += displacement * self.buoyancy_spring / 4.0

        self.velocity[None].y += total_buoyancy_force * dt
        self.velocity[None].y *= self.buoyancy_damping
        self.position[None] += self.velocity[None] * dt

        self.velocity[None].x *= self.drag
        self.velocity[None].z *= self.drag
        
        half_size_x = self.bounds[0] * 0.05
        if self.position[None].x < half_size_x:
            self.position[None].x, self.velocity[None].x = half_size_x, 0.0
        if self.position[None].x > self.bounds[0] - half_size_x:
            self.position[None].x, self.velocity[None].x = self.bounds[0] - half_size_x, 0.0

    @ti.kernel
    def control(self, move_dir: ti.types.vector(2, ti.f32), dt: ti.f32):
        self.velocity[None].x += move_dir.x * self.move_acc * dt
        self.velocity[None].z += move_dir.y * self.move_acc * dt

    @ti.kernel
    def sink(self, dt: ti.f32, water_h: ti.template()):
        if self.is_sinking[None] == 1:
            pos = self.position[None]
            grid_x, grid_z = ti.cast(pos.x, ti.i32), ti.cast(pos.z, ti.i32)
            if 0 <= grid_x < self.bounds[0] and 0 <= grid_z < self.bounds[1]:
                water_level = water_h[grid_x, grid_z]
                max_sink_y = water_level - self.max_sink_depth[None]

                if pos.y > max_sink_y:
                    self.sink_velocity[None] += 0.8 * dt
                    self.position[None].y -= self.sink_velocity[None] * dt
                else:
                    self.sink_velocity[None] = 0.0
                    self.position[None].y = max_sink_y
                    self.is_sinking[None] = 0

@ti.func
def create_rotation_matrix(rotation: ti.template()):
    pitch, yaw, roll = rotation[0], rotation[1], rotation[2]
    cos_p, sin_p = ti.cos(pitch), ti.sin(pitch)
    cos_y, sin_y = ti.cos(yaw), ti.sin(yaw)
    cos_r, sin_r = ti.cos(roll), ti.sin(roll)
    Rx = ti.Matrix([[1, 0, 0], [0, cos_p, -sin_p], [0, sin_p, cos_p]])
    Ry = ti.Matrix([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])
    Rz = ti.Matrix([[cos_r, -sin_r, 0], [sin_r, cos_r, 0], [0, 0, 1]])
    return Ry @ Rx @ Rz

Obstacle_dtype = ti.types.struct(position=ti.types.vector(3, ti.f32),velocity=ti.types.vector(3, ti.f32),size=ti.f32,active=ti.i32)

@ti.data_oriented
class ObstacleManager:
    def __init__(self, max_obstacles, spawn_area):
        self.max_obstacles = max_obstacles
        self.obstacles = Obstacle_dtype.field(shape=(self.max_obstacles,))
        self.spawn_area = spawn_area
        self.spawn_timer = 0.0
        self.group_spawn_interval = 5.0
        self.obstacles_per_group_min = 4
        self.obstacles_per_group_max = 6
        self.num_lanes = 8
        self.lane_indices_to_spawn = ti.field(ti.i32, shape=self.obstacles_per_group_max)
        self.collision_flag = ti.field(ti.i32, shape=())

    @ti.kernel
    def init_obstacles(self):
        for i in self.obstacles:
            self.obstacles[i].active = 0
            self.obstacles[i].position = ti.Vector([0.0, -1000.0, 0.0])
        self.collision_flag[None] = 0

    @ti.kernel
    def update_obstacles(self, dt: ti.f32):
        for i in self.obstacles:
            if self.obstacles[i].active == 1:
                self.obstacles[i].position += self.obstacles[i].velocity * dt
                if self.obstacles[i].position.z > self.spawn_area[1] + 20:
                    self.obstacles[i].active = 0
                    self.obstacles[i].position = ti.Vector([0.0, -1000.0, 0.0])

    def try_spawn(self, dt: ti.f32):
        self.spawn_timer += dt
        if self.spawn_timer > self.group_spawn_interval:
            self.spawn_timer = 0.0
            num_to_spawn = random.randint(self.obstacles_per_group_min, self.obstacles_per_group_max)
            all_lanes = list(range(self.num_lanes))
            random.shuffle(all_lanes)
            lanes_to_spawn_in = all_lanes[:num_to_spawn]
            for i in range(num_to_spawn):
                self.lane_indices_to_spawn[i] = lanes_to_spawn_in[i]
            self.spawn_obstacle_group_kernel(num_to_spawn)

    @ti.kernel
    def spawn_obstacle_group_kernel(self, num_to_spawn: ti.i32):
        spawn_count, obstacle_idx = 0, 0
        while spawn_count < num_to_spawn and obstacle_idx < self.max_obstacles:
            if self.obstacles[obstacle_idx].active == 0:
                self.obstacles[obstacle_idx].active = 1
                lane_idx = self.lane_indices_to_spawn[spawn_count]
                lane_width = self.spawn_area[0] / self.num_lanes
                pos_x = (lane_idx + ti.random() * 0.6 - 0.3) * lane_width
                self.obstacles[obstacle_idx].position = ti.Vector([pos_x, 10.0, ti.random() * -30.0])
                self.obstacles[obstacle_idx].velocity = ti.Vector([0.0, 0.0, 45.0])
                

                self.obstacles[obstacle_idx].size = ti.random() * 4.0 + 10.0
                # -----------------------------

                spawn_count += 1
            obstacle_idx += 1
            
    @ti.kernel
    def check_collisions(self, boat_pos: ti.types.vector(3, ti.f32), boat_radius: ti.f32):
        for i in self.obstacles:
            if self.obstacles[i].active == 1:
                dist_vec = self.obstacles[i].position - boat_pos
                dist_vec.y = 0.0
                dist_sq = dist_vec.dot(dist_vec)
                sum_radii = self.obstacles[i].size + boat_radius
                sum_radii_sq = sum_radii * sum_radii
                if dist_sq < sum_radii_sq:
                    self.collision_flag[None] = 1
                    self.obstacles[i].active = 0
                    self.obstacles[i].position = ti.Vector([0.0, -1000.0, 0.0])