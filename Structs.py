import copy
from math import *
import shapely.geometry

class Point:
    def __init__(self, x=0., y =0.):
        self.x = float(x)
        self.y = float(y)

    def set(self, x, y):
        self.x = float(x)
        self.y = float(y)

    def max(self, point):
        return Point(max(point.x, self.x), max(point.y, self.y))

    def min(self, point):
        return Point(min(point.x, self.x), min(point.y, self.y))

    def move(self, x, y):
        self.x += x
        self.y += y

    def get_values(self):
        return self.x, self.y

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __truediv__(self, other):
        return Point(self.x / other, self.y / other)

    def pp(self):
        print(self.x, self.y)


class Polygon:

    def __init__(self):
        self.points = []
        self.validSize = None
        self.ishole = None
        self.color = None

    def bounds(self):
        min_co = Point(float('inf'), float('inf'))
        max_co = Point(-float('inf'), -float('inf'))

        for point in self.points:
            min_co = point.min(min_co)
            max_co = point.max(max_co)
        return max_co, min_co

    def center(self):
        a, b = self.bounds()
        return (a + b)/2

    def add_point(self, point):
        self.points.append(copy.deepcopy(point))

    def get_points(self):
        return self.points

    def set_points(self, points):
        self.points = copy.deepcopy(points)

    def set_color(self, color):
        self.color = copy.copy(color)

    def get_color(self):
        return self.color

    def move(self, x, y):
        for point in self.points:
            point.move(x, y)

    def rotate(self, angle):

        center_point = self.center()
        self.rotate_around_point(center_point, angle * pi / 180)

    def rotate_around_point(self, center, angle):
        cx, cy = center.get_values()
        for point in self.points:
            x0, y0 = point.get_values()
            new_x = (x0 - cx) * cos(angle) - (y0 - cy) * sin(angle) + cx
            new_y = (x0 - cx) * sin(angle) + (y0 - cy) * cos(angle) + cy
            point.set(new_x, new_y)

    def shadow_poly(self):
        return shapely.geometry.Polygon([(point.x, point.y) for point in self.points])

    def is_inside_cloth(self, height):
        x_list = list()
        y_list = list()
        for point in self.points:
            x_list.append(point.x)
            y_list.append(point.y)

        if min(y_list) >= 0 and min(x_list) >= 0 and max(y_list) <= height:
            return 1
        else:
            return 0

class Reward:
    def __init__(self, reward1=0., reward2=0.):
        self.reward1 = reward1
        self.reward2 = reward2
        self.final_reward = -self.reward1 + self.reward2


class PolygonSet:
    def __init__(self, polygon_set=[]):
        self.polygon_set = polygon_set
        self.bbox =None

    def set_set(self, polygon_set):
        self.polygon_set = copy.deepcopy(polygon_set)

    def add_polygon(self, polygon):
        self.polygon_set.append(polygon)

    def area(self):
        area = 0
        for poly in self.polygon_set:
            shadow_polygon = poly.shadow_poly()
            area += shadow_polygon.area
        return area

    def get_bbox(self):
        self.calculate_bbox()
        return self.bbox

    def calculate_bbox(self):
        x_list = list()
        y_list = list()
        for poly in self.polygon_set:
            for point in poly.points:
                    x_list.append(point.x)
                    y_list.append(point.y)
        self.bbox = [Point(min(x_list), min(y_list)), Point(max(x_list), min(y_list)), Point(max(x_list), max(y_list)),
                     Point(min(x_list), max(y_list))]





class State:
    def __init__(self):
        self.polygons_src = PolygonSet()

        self.cloth_height = 500
        self.cut_num = 5
        self.max_cut_points = 10
        self.max_cut_width = 50
        self.max_cut_height = 100

        self.rgb = None

        self.reward = Reward()
        self.reward_record = Reward()
