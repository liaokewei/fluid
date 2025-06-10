import taichi as ti
import math
import random
import numpy as np

@ti.data_oriented
class WaterSurfaceSimulator:
    def __init__(self, size_x, size_z, dx=1.0):
        self.size_x = size_x
        self.size_z = size_z
        self.dx = dx

        self.dt = 0.016
        self.c = 20.0
        self.damping = 0.996 # 阻尼可以设一个稍大的值，让波纹更持久

        self.wind_direction = ti.Vector([-0.5, -0.8])
        self.wind_strength = 0.8
        self.wind_frequency = 0.05
        self.wind_speed = 5.0

        self.h = ti.field(dtype=ti.f32, shape=(self.size_x, self.size_z))
        self.v = ti.field(dtype=ti.f32, shape=(self.size_x, self.size_z))
        
        self.elapsed_time = ti.field(dtype=ti.f32, shape=())
        self.perm = ti.field(ti.i32, shape=512)
        self.init_perlin_noise()

    def init_perlin_noise(self):
        p = list(range(256))
        random.shuffle(p)
        self.perm.from_numpy(np.array(p + p, dtype=np.int32))

    @ti.kernel
    def init_height_field(self):
        self.h.fill(0.0)
        self.v.fill(0.0)
        self.elapsed_time[None] = 0.0

    # ... [fade, lerp, grad, perlin_noise, fbm 函数保持不变] ...
    @ti.func
    def fade(self, t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    @ti.func
    def lerp(self, a, b, t):
        return a + t * (b - a)

    @ti.func
    def grad(self, hash, x, y, z):
        h = hash & 15
        u = x if h < 8 else y
        v = y if h < 4 else (x if h == 12 or h == 14 else z)
        return ((u if (h & 1) == 0 else -u) +
                (v if (h & 2) == 0 else -v))

    @ti.func
    def perlin_noise(self, p):
        x, y, z = p[0], p[1], p[2]
        X = ti.cast(ti.floor(x), ti.i32) & 255
        Y = ti.cast(ti.floor(y), ti.i32) & 255
        Z = ti.cast(ti.floor(z), ti.i32) & 255
        x_frac, y_frac, z_frac = x - ti.floor(x), y - ti.floor(y), z - ti.floor(z)
        u, v, w = self.fade(x_frac), self.fade(y_frac), self.fade(z_frac)
        A = self.perm[X] + Y
        AA = self.perm[A] + Z
        AB = self.perm[A + 1] + Z
        B = self.perm[X + 1] + Y
        BA = self.perm[B] + Z
        BB = self.perm[B + 1] + Z
        g1 = self.grad(self.perm[AA], x_frac, y_frac, z_frac)
        g2 = self.grad(self.perm[BA], x_frac - 1, y_frac, z_frac)
        g3 = self.grad(self.perm[AB], x_frac, y_frac - 1, z_frac)
        g4 = self.grad(self.perm[BB], x_frac - 1, y_frac - 1, z_frac)
        g5 = self.grad(self.perm[AA + 1], x_frac, y_frac, z_frac - 1)
        g6 = self.grad(self.perm[BA + 1], x_frac - 1, y_frac, z_frac - 1)
        g7 = self.grad(self.perm[AB + 1], x_frac, y_frac - 1, z_frac - 1)
        g8 = self.grad(self.perm[BB + 1], x_frac - 1, y_frac - 1, z_frac - 1)
        return self.lerp(self.lerp(self.lerp(g1, g2, u), self.lerp(g3, g4, u), v),
                         self.lerp(self.lerp(g5, g6, u), self.lerp(g7, g8, u), v), w)

    @ti.func
    def fbm(self, p, persistence):
        total, frequency, amplitude, max_value = 0.0, 1.0, 1.0, 0.0
        for _ in ti.static(range(4)):
            total += self.perlin_noise(p * frequency) * amplitude
            max_value += amplitude
            amplitude *= persistence
            frequency *= 2.0
        return total / max_value
    
    # --- 为内部点注入背景流动能量 ---
    @ti.kernel
    def update_procedural_waves(self, time: ti.f32):
        base_amplitude = 1.0
        base_frequency = 0.02
        persistence = 0.5
        time_scale = 0.1
        flow_speed = 0.3
        for i, j in ti.ndrange((1, self.size_x - 1), (1, self.size_z - 1)):
            p = ti.Vector([i * base_frequency,
                           j * base_frequency - time * flow_speed,
                           time * time_scale])
            noise_val = self.fbm(p, persistence)
            self.v[i, j] += noise_val * base_amplitude * self.dt

    # --- 能量在内部点传播 ---
    @ti.kernel
    def wave_propagate(self):
        for i, j in ti.ndrange((1, self.size_x - 1), (1, self.size_z - 1)):
            h_laplacian = (self.h[i - 1, j] + self.h[i + 1, j] +
                           self.h[i, j - 1] + self.h[i, j + 1] -
                           4 * self.h[i, j])
            self.v[i, j] += self.dt * (self.c**2 * h_laplacian / self.dx**2)

        for i, j in ti.ndrange((1, self.size_x - 1), (1, self.size_z - 1)):
            self.h[i, j] += self.dt * self.v[i, j]
            
    # --- 新增：进行阻尼并强制处理边界 ---
    @ti.kernel
    def apply_damping_and_boundary(self):
        # 内部点施加阻尼
        for i, j in ti.ndrange((1, self.size_x - 1), (1, self.size_z - 1)):
            self.v[i, j] *= self.damping

        # 边界点强制清零
        for i in range(self.size_x):
            self.h[i, 0] = 0.0
            self.v[i, 0] = 0.0
            self.h[i, self.size_z - 1] = 0.0
            self.v[i, self.size_z - 1] = 0.0
        for j in range(self.size_z):
            self.h[0, j] = 0.0
            self.v[0, j] = 0.0
            self.h[self.size_x - 1, j] = 0.0
            self.v[self.size_x - 1, j] = 0.0

    # --- step函数采用新结构 ---
    def step(self):
        self.update_procedural_waves(self.elapsed_time[None])
        self.wave_propagate()
        self.apply_damping_and_boundary()
        self.elapsed_time[None] += self.dt

    @ti.kernel
    def disturb_at(self, x: ti.i32, z: ti.i32, radius: ti.i32, strength: ti.f32):
        for i, j in ti.ndrange((-radius, radius), (-radius, radius)):
            px, pz = x + i, z + j
            if 1 <= px < self.size_x - 1 and 1 <= pz < self.size_z - 1:
                dist_sq = i*i + j*j
                if dist_sq < radius**2:
                    # 为了让涟漪更明显，可以稍微增强这里的力量
                    falloff = ti.cos(0.5 * math.pi * ti.sqrt(dist_sq) / radius)
                    self.v[px, pz] += strength * falloff * 2.0

    @ti.kernel
    def apply_wind(self):
        # --- 修改：不再使用局部变量，而是使用 self. 属性 ---
        # wind_direction = ti.Vector([-0.5, -0.8]) # 删除这部分
        # wind_strength = 0.8
        # wind_frequency = 0.05
        # wind_speed = 5.0
        # ---------------------------------------------

        time = self.elapsed_time[None]
        wind_dir_norm = self.wind_direction.normalized()

        for i, j in ti.ndrange((1, self.size_x - 1), (1, self.size_z - 1)):
            pos_proj = i * wind_dir_norm.x + j * wind_dir_norm.y
            force = ti.sin(pos_proj * self.wind_frequency + time * self.wind_speed) * self.wind_strength

            self.v[i, j] += force * self.dt