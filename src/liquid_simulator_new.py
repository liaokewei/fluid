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

        self.h = ti.field(dtype=ti.f32, shape=(self.size_x, self.size_z))
        self.elapsed_time = ti.field(dtype=ti.f32, shape=())

        self.perm = ti.field(ti.i32, shape=512)
        self.init_perlin_noise()

    def init_perlin_noise(self):
        p = list(range(256))
        random.shuffle(p)
        self.perm.from_numpy(np.array(p + p, dtype=np.int32))

    @ti.kernel
    def init_height_field(self):
        self.h.fill(0)
        self.elapsed_time[None] = 0.0

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
        total = 0.0
        frequency = 1.0
        amplitude = 1.0
        max_value = 0.0
        for _ in ti.static(range(4)):
            total += self.perlin_noise(p * frequency) * amplitude
            max_value += amplitude
            amplitude *= persistence
            frequency *= 2.0
        return total / max_value

    @ti.kernel
    def update_procedural_waves(self, time: ti.f32):
        # --- 最终版参数，用于生成平滑、舒展的波浪 ---
        # 1. 降低整体浪高
        base_amplitude = 1.0
        # 2. 大幅降低基础频率，让波浪非常宽阔
        base_frequency = 0.01
        # 3. 降低细节强度，让表面更平滑
        persistence = 0.4
        
        time_scale = 0.2
        flow_speed = 0.3
        # ---------------------------------------------

        for i, j in self.h:
            p = ti.Vector([i * base_frequency, 
                           j * base_frequency - time * flow_speed, 
                           time * time_scale])
            noise_val = self.fbm(p, persistence)
            self.h[i, j] = noise_val * base_amplitude

    def step(self):
        self.elapsed_time[None] += self.dt
        self.update_procedural_waves(self.elapsed_time[None])
        
    @ti.kernel
    def disturb_at(self, x: ti.i32, z: ti.i32, radius: ti.i32, strength: ti.f32):
        for i, j in ti.ndrange((-radius, radius), (-radius, radius)):
            if 0 <= x + i < self.size_x and 0 <= z + j < self.size_z:
                if i**2 + j**2 < radius**2:
                    dist_sq = i*i + j*j
                    falloff = 1.0 - dist_sq / (radius*radius)
                    ti.atomic_add(self.h[x + i, z + j], strength * falloff)