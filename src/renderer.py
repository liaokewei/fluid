import taichi as ti
import taichi.math as tm
import numpy as np

def load_obj(filename):
    # ... load_obj 函数不变 ...
    vertices, normals, faces = [], [], []
    try:
        with open(filename, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    vertices.append([float(x) for x in line.strip().split()[1:]])
                elif line.startswith('vn '):
                    normals.append([float(x) for x in line.strip().split()[1:]])
                elif line.startswith('f '):
                    face_verts = [int(p.split('/')[0]) - 1 for p in line.strip().split()[1:]]
                    faces.append(face_verts)
    except FileNotFoundError:
        print(f"Error: Stone model file not found at {filename}")
        return None, None, None
    if not normals:
        print("Warning: Model has no normals. Generating simple normals.")
        norms_np = np.zeros_like(vertices)
        for i, face in enumerate(faces):
             v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
             normal = np.cross(np.subtract(v1, v0), np.subtract(v2, v0))
             norm_val = np.linalg.norm(normal)
             if norm_val > 0: normal /= norm_val
             for vid in face: norms_np[vid] += normal
        for i in range(len(norms_np)):
            norm_val = np.linalg.norm(norms_np[i])
            if norm_val > 0: norms_np[i] /= norm_val
        normals = norms_np.tolist()
    triangles = []
    for face in faces:
        if len(face) == 3: triangles.append(face)
        elif len(face) == 4:
            triangles.append([face[0], face[1], face[2]])
            triangles.append([face[0], face[2], face[3]])
    return np.array(vertices, dtype=np.float32), np.array(normals, dtype=np.float32), np.array(triangles, dtype=np.int32)


@ti.data_oriented
class WaterRenderer:
    def __init__(self, water_sim, boat, obstacle_manager, smoke_system):
        self.water_sim = water_sim
        self.boat = boat
        self.obstacle_manager = obstacle_manager
        self.smoke_system = smoke_system

        self.visual_height_scale = 15.0
        # ... 其他参数不变 ...
        self.color_contrast = 2.0
        self.crest_color = ti.Vector([0.6, 0.8, 1.0])
        self.trough_color = ti.Vector([0.2, 0.45, 0.75])
        self.boat_color = (0.45, 0.33, 0.22)
        self.specular_strength = 0.8
        self.shininess = 128
        self.obstacle_color = (0.3, 0.3, 0.35)
        self.fresnel_strength = 0.6
        self.sky_color = ti.Vector([0.6, 0.75, 0.9])
        self.smoke_color = (0.5, 0.5, 0.5)

        # 水面渲染数据 (不变)
        num_vertices = self.water_sim.size_x * self.water_sim.size_z
        self.water_vertices = ti.Vector.field(3, dtype=ti.f32, shape=num_vertices)
        self.water_vertex_normals = ti.Vector.field(3, dtype=ti.f32, shape=num_vertices)
        self.water_vertex_colors = ti.Vector.field(3, dtype=ti.f32, shape=num_vertices)
        num_triangles = (self.water_sim.size_x - 1) * (self.water_sim.size_z - 1) * 2
        self.water_indices = ti.field(dtype=ti.i32, shape=num_triangles * 3)
        self.init_water_indices()

        # 石块模型基础数据 (不变)
        stone_verts_np, stone_norms_np, stone_faces_np = load_obj('../model/stone.obj')
        if stone_verts_np is None: raise SystemExit("Renderer failed to load stone model.")
        num_stone_verts = stone_verts_np.shape[0]
        self.stone_base_vertices = ti.Vector.field(3, dtype=ti.f32, shape=num_stone_verts)
        self.stone_base_normals = ti.Vector.field(3, dtype=ti.f32, shape=num_stone_verts)
        self.stone_indices = ti.field(dtype=ti.i32, shape=stone_faces_np.flatten().shape[0])
        self.stone_base_vertices.from_numpy(stone_verts_np)
        self.stone_base_normals.from_numpy(stone_norms_np)
        self.stone_indices.from_numpy(stone_faces_np.flatten())
        
        # --- 优化 1: 创建一个能容纳所有石块顶点/法线数据的大缓冲区 ---
        max_obs = self.obstacle_manager.max_obstacles
        self.all_obstacles_vertices = ti.Vector.field(3, dtype=ti.f32, shape=(max_obs, num_stone_verts))
        self.all_obstacles_normals = ti.Vector.field(3, dtype=ti.f32, shape=(max_obs, num_stone_verts))
        # -----------------------------------------------------------

        # 单个石块的渲染缓冲区 (保持不变, 用于提交给 scene.mesh)
        self.obstacle_render_buffer = ti.Vector.field(3, dtype=ti.f32, shape=num_stone_verts)
        self.obstacle_normal_buffer = ti.Vector.field(3, dtype=ti.f32, shape=num_stone_verts)


    @ti.kernel
    def init_water_indices(self):
        # ... 不变 ...
        for i, j in ti.ndrange(self.water_sim.size_x - 1, self.water_sim.size_z - 1):
            quad_id = (i * (self.water_sim.size_z - 1)) + j
            v_idx_00 = i * self.water_sim.size_z + j; v_idx_10 = (i + 1) * self.water_sim.size_z + j
            v_idx_01 = i * self.water_sim.size_z + (j + 1); v_idx_11 = (i + 1) * self.water_sim.size_z + (j + 1)
            self.water_indices[quad_id * 6 + 0] = v_idx_00; self.water_indices[quad_id * 6 + 1] = v_idx_10
            self.water_indices[quad_id * 6 + 2] = v_idx_01; self.water_indices[quad_id * 6 + 3] = v_idx_10
            self.water_indices[quad_id * 6 + 4] = v_idx_11; self.water_indices[quad_id * 6 + 5] = v_idx_01

    @ti.kernel
    def update_water_mesh_and_color(self, camera_pos: ti.math.vec3, light_pos: ti.math.vec3):
        # ... 不变 ...
        for i, j in ti.ndrange(self.water_sim.size_x, self.water_sim.size_z):
            idx = i * self.water_sim.size_z + j
            world_pos = ti.Vector([i, self.water_sim.h[i, j], j]) * ti.Vector([1.0, self.visual_height_scale, 1.0])
            self.water_vertices[idx] = world_pos
            h_xp = self.water_sim.h[min(i + 1, self.water_sim.size_x - 1), j]; h_xm = self.water_sim.h[max(i - 1, 0), j]
            h_zp = self.water_sim.h[i, min(j + 1, self.water_sim.size_z - 1)]; h_zm = self.water_sim.h[i, max(j - 1, 0)]
            normal = ti.Vector([h_xm - h_xp, 2.0 / self.visual_height_scale, h_zm - h_zp]).normalized()
            self.water_vertex_normals[idx] = normal
            height = self.water_sim.h[i, j]; normalized_h = height * self.color_contrast * 0.5 + 0.5
            t = tm.clamp(normalized_h, 0.0, 1.0); refraction_color = self.crest_color * t + self.trough_color * (1.0 - t)
            view_dir = (camera_pos - world_pos).normalized(); f0 = 0.04
            fresnel_factor = f0 + (1.0 - f0) * tm.pow(1.0 - max(0.0, view_dir.dot(normal)), 5.0)
            fresnel_factor *= self.fresnel_strength; base_color = tm.mix(refraction_color, self.sky_color, fresnel_factor)
            light_dir = (light_pos - world_pos).normalized(); half_dir = (view_dir + light_dir).normalized()
            specular_intensity = tm.pow(max(0.0, normal.dot(half_dir)), self.shininess)
            specular_color = self.specular_strength * specular_intensity * ti.Vector([1.0, 1.0, 1.0])
            final_color = base_color + specular_color
            self.water_vertex_colors[idx] = final_color

    # --- 优化 2: 创建一个“大内核”一次性更新所有石块的变换 ---
    @ti.kernel
    def _update_all_stones_transforms(self):
        # 这个内核在 2D 网格上并行，每个线程负责一个石块的一个顶点
        for i, j in ti.ndrange(self.obstacle_manager.max_obstacles, self.stone_base_vertices.shape[0]):
            # 检查这个石块是否是激活状态
            if self.obstacle_manager.obstacles[i].active == 1:
                obs_data = self.obstacle_manager.obstacles[i]
                pos = obs_data.position
                size = obs_data.size

                # 计算石块在水面上的位置
                grid_x = max(0, min(self.water_sim.size_x - 1, int(pos.x)))
                grid_z = max(0, min(self.water_sim.size_z - 1, int(pos.z)))
                water_y = self.water_sim.h[grid_x, grid_z] * self.visual_height_scale
                world_pos = ti.Vector([pos.x, water_y, pos.z])

                # 变换
                angle = 1.5; c, s = ti.cos(angle), ti.sin(angle)
                rot_matrix = ti.Matrix([[c, 0, s], [0, 1, 0], [-s, 0, c]])
                
                transformed_vert = (rot_matrix @ self.stone_base_vertices[j]) * size + world_pos
                transformed_norm = rot_matrix @ self.stone_base_normals[j]
                
                # 将结果存入大缓冲区
                self.all_obstacles_vertices[i, j] = transformed_vert
                self.all_obstacles_normals[i, j] = transformed_norm

    # --- 优化 3: 创建一个轻量级内核，用于将数据从大缓冲区拷贝到渲染缓冲区 ---
    @ti.kernel
    def _copy_stone_data_to_buffer(self, stone_idx: int):
        num_verts = self.stone_base_vertices.shape[0]
        for i in range(num_verts):
            self.obstacle_render_buffer[i] = self.all_obstacles_vertices[stone_idx, i]
            self.obstacle_normal_buffer[i] = self.all_obstacles_normals[stone_idx, i]

    def render(self, scene, game_over: bool, camera_pos: ti.math.vec3, light_pos: ti.math.vec3, sinking_flag: int = 0):
        # ... 渲染水面和船 (不变) ...
        self.update_water_mesh_and_color(camera_pos, light_pos)
        scene.mesh(self.water_vertices, indices=self.water_indices, per_vertex_color=self.water_vertex_colors, normals=self.water_vertex_normals, two_sided=True)
        if not game_over or sinking_flag == 1:
            scene.mesh(vertices=self.boat.world_vertices, indices=self.boat.indices, normals=self.boat.world_normals, color=self.boat_color,per_vertex_color=self.boat.vertex_color)

        # 渲染石块障碍物
        if self.obstacle_manager.max_obstacles > 0:
            # --- 优化 4: 修改渲染循环 ---
            # 1. 在循环外，一次性调用大内核计算所有石块的位置
            self._update_all_stones_transforms()

            # 2. 在循环内，只做轻量级的数据拷贝和绘制调用
            for i in range(self.obstacle_manager.max_obstacles):
                if self.obstacle_manager.obstacles[i].active == 1:
                    # 将第 i 个石块的数据拷贝到渲染缓冲区
                    self._copy_stone_data_to_buffer(i)
                    
                    # 使用缓冲区进行绘制
                    scene.mesh(
                        vertices=self.obstacle_render_buffer,
                        indices=self.stone_indices,
                        normals=self.obstacle_normal_buffer,
                        color=self.obstacle_color
                    )
        
        # 渲染烟雾 (不变)
        scene.particles(self.smoke_system.particles.position, radius=0.3, color=self.smoke_color)