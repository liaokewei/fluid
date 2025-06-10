import taichi as ti
import taichi.math as tm
import numpy as np

@ti.data_oriented
class WaterRenderer:
    def __init__(self, water_sim, boat, obstacle_manager, smoke_system):
        self.water_sim = water_sim
        self.boat = boat
        self.obstacle_manager = obstacle_manager
        self.smoke_system = smoke_system

        # --- 从 water_sim 和 boat 对象中获取视觉参数 ---
        # 这种方式更灵活，我们可以在gui.py中统一调整这些参数
        self.visual_height_scale = 15.0 # 你可以随时在这里调整
        self.color_contrast = 2.0
        self.crest_color = ti.Vector([0.6, 0.8, 1.0])
        self.trough_color = ti.Vector([0.2, 0.45, 0.75])
        self.boat_color = (0.45, 0.33, 0.22)
        self.specular_strength = 0.8  # 高光强度
        self.shininess = 128
        self.obstacle_color = (0.9, 0.9, 0.95)

        self.fresnel_strength = 0.6  # 菲涅尔效应的强度
        self.sky_color = ti.Vector([0.6, 0.75, 0.9]) # 天空的颜色

        self.smoke_color = (0.5, 0.5, 0.5)

        # --- 核心修正：使用一维 Field 存储顶点数据 ---
        num_vertices = self.water_sim.size_x * self.water_sim.size_z
        self.water_vertices = ti.Vector.field(3, dtype=ti.f32, shape=num_vertices)
        self.water_vertex_normals = ti.Vector.field(3, dtype=ti.f32, shape=num_vertices)
        self.water_vertex_colors = ti.Vector.field(3, dtype=ti.f32, shape=num_vertices)

        # --- 优化：在初始化时一次性计算好索引 ---
        num_triangles = (self.water_sim.size_x - 1) * (self.water_sim.size_z - 1) * 2
        self.water_indices = ti.field(dtype=ti.i32, shape=num_triangles * 3)
        self.init_water_indices()

    @ti.kernel
    def init_water_indices(self):
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

    @ti.kernel
    def update_water_mesh_and_color(self, camera_pos: ti.math.vec3, light_pos: ti.math.vec3):
        for i, j in ti.ndrange(self.water_sim.size_x, self.water_sim.size_z):
            # ... [第1、2步计算坐标和法线，保持不变] ...
            idx = i * self.water_sim.size_z + j
            world_pos = ti.Vector([i, self.water_sim.h[i, j], j])
            world_pos *= ti.Vector([1.0, self.visual_height_scale, 1.0])
            self.water_vertices[idx] = world_pos
            h_c = self.water_sim.h[i, j]
            h_xp = self.water_sim.h[min(i + 1, self.water_sim.size_x - 1), j]
            h_xm = self.water_sim.h[max(i - 1, 0), j]
            h_zp = self.water_sim.h[i, min(j + 1, self.water_sim.size_z - 1)]
            h_zm = self.water_sim.h[i, max(j - 1, 0)]
            normal = ti.Vector([h_xm - h_xp, 2.0 / self.visual_height_scale, h_zm - h_zp]).normalized()
            self.water_vertex_normals[idx] = normal

            # 3. 计算水体本身颜色（折射/漫反射）
            height = self.water_sim.h[i, j]
            normalized_h = height * self.color_contrast * 0.5 + 0.5
            t = tm.clamp(normalized_h, 0.0, 1.0)
            refraction_color = self.crest_color * t + self.trough_color * (1.0 - t)

            # --- 4. 新增：计算菲涅尔效应 ---
            view_dir = (camera_pos - world_pos).normalized()

            # 使用Schlick近似计算菲涅尔系数
            f0 = 0.04 # 适用于水等电介质的基础反射率
            fresnel_factor = f0 + (1.0 - f0) * tm.pow(1.0 - max(0.0, view_dir.dot(normal)), 5.0)
            fresnel_factor *= self.fresnel_strength

            # 根据菲涅尔效应，混合水体颜色和天空反射颜色
            base_color = tm.mix(refraction_color, self.sky_color, fresnel_factor)
            # -------------------------------

            # 5. 计算镜面高光
            light_dir = (light_pos - world_pos).normalized()
            half_dir = (view_dir + light_dir).normalized()
            specular_intensity = tm.pow(max(0.0, normal.dot(half_dir)), self.shininess)
            specular_color = self.specular_strength * specular_intensity * ti.Vector([1.0, 1.0, 1.0])

            # 6. 混合最终颜色
            final_color = base_color + specular_color
            self.water_vertex_colors[idx] = final_color

    def render(self, scene, game_over: bool, camera_pos: ti.math.vec3, light_pos: ti.math.vec3, sinking_flag: int = 0):
        # 1. 更新水面数据
        self.update_water_mesh_and_color(camera_pos, light_pos)

        # 2. 渲染水面
        scene.mesh(self.water_vertices,
               indices=self.water_indices,
               per_vertex_color=self.water_vertex_colors,
               normals=self.water_vertex_normals,
               two_sided=True)

        # 3. 渲染其他物体
        if not game_over or sinking_flag == 1:
            # 使用 world_vertices 来渲染移动和旋转后的船
            scene.mesh(vertices=self.boat.world_vertices, 
                       indices=self.boat.indices, 
                       normals=self.boat.world_normals, 
                       color=self.boat_color,per_vertex_color=self.boat.vertex_color)
            # 使用 scene.particles 渲染障碍物
            scene.particles(self.obstacle_manager.obstacles.position, 
                            radius=6.0, 
                            color=self.obstacle_color)
        
        # 使用正确的字段渲染烟雾
        scene.particles(self.smoke_system.particles.position,
                        radius=0.3,
                        color=self.smoke_color)