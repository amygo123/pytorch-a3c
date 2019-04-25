from Structs import *
import random
from random import randint
import copy
import shapely
from shapely.ops import *
from shapely.geometry import mapping
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Env:
    def __init__(self, random_seed = None):
        self.state = State()
        self.last_state = None
        self.state_backup = None
        if random_seed != None:
            np.random.seed(random_seed)
        self.action_space = 30

    def reset(self):
        self.init_polygons()
        return self.get_state_array()

    def init_polygons(self, cut_num=5, max_cut_points=10, max_cut_width=50, max_cut_height=100, cloth_height=500):
        self.state.cut_num = cut_num
        self.state.max_cut_points = max_cut_points
        self.state.max_cut_width = max_cut_width
        self.state.max_cut_height = max_cut_height
        self.state.cloth_height = cloth_height
        if cut_num < 2:
            return 0
        polygon_list = []
        for i in range(0, cut_num):
            flag = 1

            while flag:

                points = [Point(np.random.rand() * float(max_cut_width), np.random.rand() * float(max_cut_height)) for _ in range(0, max_cut_points)]
                poly = shapely.geometry.Polygon(
                    [(point.x, point.y) for point in points])
                if poly.is_valid:
                    polygon = Polygon()
                    polygon.set_points(points)
                    polygon_list.append(polygon)
                    flag = 0
        self.state.polygons_src.set_set(polygon_list)
        self.calculate_reward()
        self.state.rgb = np.random.random((self.state.cut_num, 3))
        self.last_state = copy.deepcopy(self.state)

    def init_last_state(self):
        self.state = copy.deepcopy(self.last_state)

    def init_same_polygon_shapes(self):
        self.state = copy.deepcopy(self.last_state)

        for i in range(self.state.cut_num):
            dx = random.random() * self.state.cloth_height
            dy = random.random() * self.state.cloth_height * 1/2
            dangle = random.random() * 360
            self.action(dx, dy, dangle, i)
            # self.action(100, 100, 90, i)

    def action(self, dx, dy, dangle, p_id):
        self.state_backup = copy.deepcopy(self.state)

        self.state.polygons_src.polygon_set[p_id].move(dx, dy)
        self.state.polygons_src.polygon_set[p_id].rotate(dangle)
        self.calculate_reward()
        inside = self.state.polygons_src.polygon_set[p_id].is_inside_cloth(self.state.cloth_height)
        if not inside:
            self.state = copy.deepcopy(self.state_backup)

    def step(self, action):


        unit = 5
        dx = 0
        dy = 0
        dangle = 0
        a_ind = action % 5
        if a_ind == 0:
            dx = unit
        elif a_ind == 1:
            dx = -unit
        elif a_ind == 2:
            dy = unit
        elif a_ind == 3:
            dy = -unit
        elif a_ind == 4:
            dangle = unit
        else:
            dangle = -unit

        id = int(action / 6)
        self.action(dx, dy, dangle, id)
        reward = self.give_reward()
        done = self.state.reward.reward1 == 1 and self.state.reward.reward2 >= 0.9
        if done:
            reward += 1
        return self.get_state_array(), reward, done


    def give_reward(self):
        rew = 0
        if self.state.reward.reward1 == 1 and self.state.reward_record.reward1 < 1:
            rew += 0.5
        if self.state.reward.reward1 >= self.state.reward_record.reward1 and \
            self.state.reward.reward2 > self.state.reward_record.reward2:
            rew += 0.01
            self.state.reward_record.reward1 = self.state.reward.reward1
            self.state.reward_record.reward2 = self.state.reward.reward2

        return rew



    def calculate_union_area(self):
        polys = []
        for polygon in self.state.polygons_src.polygon_set:
            poly = polygon.shadow_poly()
            polys.append(poly)
        union_poly = unary_union(polys)
        return union_poly.area

    def calculate_reward(self):
        union_area = self.calculate_union_area()
        self.state.reward.reward1 = union_area / self.state.polygons_src.area()

        bound_points = self.state.polygons_src.get_bbox()
        areaCloth = self.state.cloth_height * (bound_points[1].x - bound_points[0].x)
        self.state.reward.reward2 = union_area / areaCloth
        self.state.reward.final_reward = self.state.reward.reward1 + self.state.reward.reward2

    def get_state_array(self):
        value_list = []
        for poly in self.state.polygons_src.polygon_set:
            for point in poly.points:
                value_list.append(point.x)
                value_list.append(point.y)
        out = np.array(value_list) / self.state.cloth_height
        return out

    def get_polygon(self):
        coor = np.zeros(shape=[self.state.cut_num, self.state.max_cut_points, 2], dtype=float)
        pi=-1
        for poly in self.state.polygons_src.polygon_set:
            pi+=1
            ppi = -1

            # color = np.array([randint(0, 255), randint(0, 255), randint(0, 255)])
            for point in poly.points:
                ppi += 1

                coor[pi,ppi,0] = point.x
                coor[pi,ppi,1] = point.y
        return coor, self.state.polygons_src.get_bbox()[2].x, self.state.cloth_height

    def visual(self):
        cor, xlim, ylim = self.get_polygon()
        from matplotlib.collections import PolyCollection
        fig = plt.figure()
        ax = fig.add_subplot(211, aspect='auto')
        lim = max(xlim, ylim)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.invert_xaxis()
        ax.invert_yaxis()
        polys = PolyCollection(cor, edgecolor=self.state.rgb, facecolors='None')
        ax.add_collection(polys)
        plt.show()


if __name__ == '__main__':
    e = Env()
    e.init_polygons()
    # e.init_same_polygon_shapes()
    # e.init_last_state()
    # e.init_same_polygon_shapes()
    # e.init_last_state()
    # e.init_same_polygon_shapes()
    li = e.get_state_array()
    # from matplotlib.collections import PolyCollection
    #
    # _, cor, xlim, ylim= e.get_state_array()
    # # ax.imshow(pic)
    # fig = plt.figure()
    # ax = fig.add_subplot(211, aspect='auto')
    # xy = cor[0, :, :]
    # lim =max(xlim,ylim)
    # ax.set_xlim(lim)
    # ax.set_ylim(lim)
    # ax.invert_xaxis()
    # ax.invert_yaxis()
    # rgb = np.random.random((xy.shape[0], 3))
    # poly = plt.Polygon(xy)
    # polys = PolyCollection(cor, edgecolor = rgb, facecolors ='None')
    # ax.add_collection(polys)
    # plt.show()

    for i in range(100):
        # action = [random.random() * 0.1, random.random() * 0.1, random.random() * np.pi, randint(0, 4)]
        _, reward, done = e.step(13)
        print(reward)
        e.visual()







