import taichi as ti
import numpy as np

def load_obj(filename):
    # ... load_obj 函数保持不变 ...
    vertices = []
    normals = []
    faces = []
    face_normals_idx = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    vertices.append([float(x) for x in line.strip().split()[1:]])
                elif line.startswith('vn '):
                    normals.append([float(x) for x in line.strip().split()[1:]])
                elif line.startswith('f '):
                    face_verts, face_norms = [], []
                    parts = line.strip().split()[1:]
                    for part in parts:
                        v, vt, vn = (part.split('/') + [None, None])[:3]
                        face_verts.append(int(v) - 1)
                        if vn is not None: face_norms.append(int(vn) - 1)
                    faces.append(face_verts)
                    if face_norms: face_normals_idx.append(face_norms)
    except FileNotFoundError:
        print(f"Error: Model file not found at {filename}")
        return None, None, None
    triangles = []
    for face in faces:
        if len(face) == 3: triangles.append(face)
        elif len(face) == 4:
            triangles.append([face[0], face[1], face[2]])
            triangles.append([face[0], face[2], face[3]])
    return np.array(vertices, dtype=np.float32), np.array(normals, dtype=np.float32), np.array(triangles, dtype=np.int32)

@ti.data_oriented
class ObstacleManager:
    def __init__(self, model_path, num_obstacles=15, water_bounds=(256, 256)):
        self.num_obstacles = num_obstacles
        self.bounds = water_bounds
        
        verts_np, norms_np, faces_np = load_obj(model_path)
        if verts_np is None: raise SystemExit("Failed to load obstacle model.")
        if norms_np.shape[0] == 0: raise SystemExit(f"Model {model_path} does not contain vertex normals (vn lines).")
        
        self.num_vertices_per_obstacle = verts_np.shape[0]

        self.vertices = ti.Vector.field(3, dtype=ti.f32, shape=self.num_vertices_per_obstacle)
        self.local_normals = ti.Vector.field(3, dtype=ti.f32, shape=self.num_vertices_per_obstacle)
        self.faces = ti.Vector.field(3, dtype=ti.i32, shape=faces_np.shape[0])
        
        self.vertices.from_numpy(verts_np)
        self.local_normals.from_numpy(norms_np)
        self.faces.from_numpy(faces_np)

        self.positions = ti.Vector.field(3, dtype=ti.f32, shape=self.num_obstacles)
        self.prev_positions = ti.Vector.field(3, dtype=ti.f32, shape=self.num_obstacles)
        self.velocities = ti.Vector.field(3, dtype=ti.f32, shape=self.num_obstacles)
        self.rotations = ti.Vector.field(3, dtype=ti.f32, shape=self.num_obstacles)
        
        self.buoyancy_points = ti.Vector.field(3, dtype=ti.f32, shape=4)
        buoyancy_points_np = np.array([ [0, 0, -0.5], [0, 0, 0.5], [-0.5, 0, 0], [0.5, 0, 0] ], dtype=np.float32)
        self.buoyancy_points.from_numpy(buoyancy_points_np)

        self.scales = ti.field(dtype=ti.f32, shape=self.num_obstacles)
        self.colors = ti.Vector.field(3, dtype=ti.f32, shape=self.num_obstacles)
        self.mass = 15.0
        self.drag = 0.98
        self.buoyancy_spring = 200.0
        self.buoyancy_damping = 0.95
        self.water_force_multiplier = 40.0
        self.visual_height_scale = 15.0

    @ti.kernel
    def init_obstacles(self):
        # ... init_obstacles 内核保持不变 ...
        for i in range(self.num_obstacles):
            pos = ti.Vector([ ti.random() * self.bounds[0], 20.0, ti.random() * self.bounds[1] ])
            self.positions[i] = pos
            self.prev_positions[i] = pos
            self.velocities[i].fill(0.0)
            self.scales[i] = ti.random() * 2.0 + 4.0
            gray_val = ti.random() * 0.2 + 0.25
            self.colors[i] = ti.Vector([gray_val, gray_val, gray_val])
            self.rotations[i] = ti.Vector([0.0, ti.random() * 2 * 3.14159, 0.0])

    @ti.func
    def create_rotation_matrix(self, rotation: ti.template()):
        # ... create_rotation_matrix 函数保持不变 ...
        pitch, yaw, roll = rotation[0], rotation[1], rotation[2]
        cos_p, sin_p = ti.cos(pitch), ti.sin(pitch)
        cos_y, sin_y = ti.cos(yaw), ti.sin(yaw)
        cos_r, sin_r = ti.cos(roll), ti.sin(roll)
        Rx = ti.Matrix([[1, 0, 0], [0, cos_p, -sin_p], [0, sin_p, cos_p]])
        Ry = ti.Matrix([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])
        Rz = ti.Matrix([[cos_r, -sin_r, 0], [sin_r, cos_r, 0], [0, 0, 1]])
        return Ry @ Rx @ Rz

    @ti.kernel
    def step(self, dt: ti.f32, water_h: ti.template()):
        # ... step 内核保持不变 ...
        for i in range(self.num_obstacles):
            self.prev_positions[i] = self.positions[i]
            total_buoyancy_force = 0.0
            rot_matrix = self.create_rotation_matrix(self.rotations[i])
            for j in ti.static(range(4)):
                point_local = self.buoyancy_points[j] * self.scales[i]
                point_world = self.positions[i] + rot_matrix @ point_local
                grid_x = ti.max(0, ti.min(self.bounds[0] - 1, ti.cast(point_world.x, ti.i32)))
                grid_z = ti.max(0, ti.min(self.bounds[1] - 1, ti.cast(point_world.z, ti.i32)))
                water_level = water_h[grid_x, grid_z] * self.visual_height_scale
                displacement = water_level - point_world.y
                total_buoyancy_force += displacement * self.buoyancy_spring / 4.0
            self.velocities[i].y += total_buoyancy_force * dt / self.mass
            self.velocities[i].y *= self.buoyancy_damping
            pos = self.positions[i]
            x, z = ti.cast(pos.x, ti.i32), ti.cast(pos.z, ti.i32)
            h_xp = water_h[ti.max(0, ti.min(self.bounds[0] - 1, x + 1)), z]
            h_xm = water_h[ti.max(0, ti.min(self.bounds[0] - 1, x - 1)), z]
            h_zp = water_h[x, ti.max(0, ti.min(self.bounds[1] - 1, z + 1))]
            h_zm = water_h[x, ti.max(0, ti.min(self.bounds[1] - 1, z - 1))]
            water_flow_force = ti.Vector([(h_xm - h_xp), 0.0, (h_zm - h_zp)])
            self.velocities[i] += water_flow_force * self.water_force_multiplier * dt / self.mass
            self.velocities[i].x *= self.drag
            self.velocities[i].z *= self.drag
            self.positions[i] += self.velocities[i] * dt
            if self.positions[i].x < 0: self.positions[i].x, self.velocities[i].x = 0, -self.velocities[i].x * 0.5
            if self.positions[i].x > self.bounds[0]: self.positions[i].x, self.velocities[i].x = self.bounds[0], -self.velocities[i].x * 0.5
            if self.positions[i].z < 0: self.positions[i].z, self.velocities[i].z = 0, -self.velocities[i].z * 0.5
            if self.positions[i].z > self.bounds[1]: self.positions[i].z, self.velocities[i].z = self.bounds[1], -self.velocities[i].z * 0.5

    # --- 关键修改：移除 get_transformed_mesh_data 函数 ---
    # def get_transformed_mesh_data(self, i):
    #     ...
    
    # --- 新增 Taichi Kernel 用于变换 ---
    @ti.kernel
    def transform_kernel(self, obstacle_idx: int,
                         out_verts: ti.template(),
                         out_norms: ti.template()):
        # 获取单个石块的属性
        pos = self.positions[obstacle_idx]
        rot = self.rotations[obstacle_idx]
        scale = self.scales[obstacle_idx]
        
        rot_matrix = self.create_rotation_matrix(rot)
        
        # 并行计算每个顶点的世界坐标和法线
        for i in range(self.num_vertices_per_obstacle):
            # 变换顶点
            transformed_vert = (rot_matrix @ self.vertices[i]) * scale + pos
            out_verts[i] = transformed_vert
            
            # 变换法线 (只应用旋转)
            transformed_norm = rot_matrix @ self.local_normals[i]
            out_norms[i] = transformed_norm.normalized()