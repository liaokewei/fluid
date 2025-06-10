import taichi as ti
import random

@ti.data_oriented
class RainSimulator:
    def __init__(self, water_sim, probability, strength, radius):
        """
        雨滴模拟器
        :param water_sim: 水面模拟器的实例
        :param probability: 每一帧产生一个雨滴的概率 (0.0 到 1.0)
        :param strength: 雨滴产生的涟漪强度
        :param radius: 雨滴产生的涟漪半径
        """
        self.water_sim = water_sim
        self.probability = probability
        self.strength = strength
        self.radius = radius

    def step(self):
        # 根据概率决定本帧是否产生雨滴
        if random.random() < self.probability:
            # 在水面内部随机选择一个点
            rand_x = int(random.random() * (self.water_sim.size_x - 2)) + 1
            rand_z = int(random.random() * (self.water_sim.size_z - 2)) + 1
            
            # 在该点产生一个扰动（涟漪）
            self.water_sim.disturb_at(rand_x, rand_z, self.radius, self.strength)