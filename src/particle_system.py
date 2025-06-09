import taichi as ti

Particle_dtype = ti.types.struct(
    position=ti.types.vector(3, ti.f32),
    velocity=ti.types.vector(3, ti.f32),
    color=ti.types.vector(4, ti.f32),
    life=ti.f32,
    active=ti.i32
)

@ti.data_oriented
class ParticleSystem:
    def __init__(self, max_particles):
        self.max_particles = max_particles
        self.particles = Particle_dtype.field(shape=(self.max_particles,))
        # --- 核心修正：减小向上的力 ---
        self.gravity = ti.Vector([0.0, 8.0, 0.0]) # 原来是 20.0
        # ---------------------------

    @ti.kernel
    def init_particles(self):
        for i in self.particles:
            self.particles[i].active = 0
            self.particles[i].position = ti.Vector([0.0, -1000.0, 0.0])

    def spawn(self, pos):
        self.spawn_kernel(pos)

    @ti.kernel
    def spawn_kernel(self, pos: ti.types.vector(3, ti.f32)):
        for i in range(self.max_particles):
             if self.particles[i].active == 0:
                self.particles[i].position = pos + ti.Vector([
                    (ti.random() - 0.5) * 20, 
                    (ti.random() - 0.5) * 10,
                    (ti.random() - 0.5) * 20
                ])
                # --- 核心修正：减小初始速度 ---
                self.particles[i].velocity = ti.Vector([
                    (ti.random() - 0.5) * 5,  # 原来是 15
                    ti.random() * 10 + 10,  # 原来是 20 + 20
                    (ti.random() - 0.5) * 5   # 原来是 15
                ])
                # -----------------------------
                self.particles[i].life = ti.random() * 1.5 + 1.5 # 寿命1.5-3秒
                self.particles[i].color = ti.Vector([0.5, 0.5, 0.5, 1.0])
                self.particles[i].active = 1
    
    @ti.kernel
    def update(self, dt: ti.f32):
        for i in self.particles:
            if self.particles[i].active == 1:
                self.particles[i].velocity += self.gravity * dt
                self.particles[i].position += self.particles[i].velocity * dt
                self.particles[i].life -= dt
                
                if self.particles[i].life > 0:
                    # 将最大寿命从 2 改为 3
                    self.particles[i].color[3] = self.particles[i].life / 3.0 
                else:
                    self.particles[i].color[3] = 0.0

                if self.particles[i].life < 0:
                    self.particles[i].active = 0
                    self.particles[i].position = ti.Vector([0.0, -1000.0, 0.0])