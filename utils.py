import numpy as np
import math

def angle(pt1, pt2, pt0):
    dx1 = pt1[0] - pt0[0]
    dy1 = pt1[1] - pt0[1]
    dx2 = pt2[0] - pt0[0]
    dy2 = pt2[1] - pt0[1]
    angle = math.degrees(math.acos(
        (dx1 * dx2 + dy1 * dy2) /
        (math.sqrt((dx1 ** 2 + dy1 ** 2) * (dx2 ** 2 + dy2 ** 2)) + 1e-10)
    ))
    return angle
