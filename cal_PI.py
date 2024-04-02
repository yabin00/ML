import math

def calculate_pi(iterations):
    pi = 0
    sign = 1

    for i in range(iterations):
        denominator = 2 * i + 1
        term = sign * (1 / denominator)
        pi += term
        sign *= -1
        
    pi *= 4
    return pi

# 设置迭代次数来控制近似精度
iterations = 10000000
approx_pi = calculate_pi(iterations)

print(f"Approximation of pi: {approx_pi}")
print(f"Error: {math.pi - approx_pi}")

import random

def calculate_pi(num_points):
    points_inside_circle = 0
    total_points = num_points

    for _ in range(num_points):
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        distance = x**2 + y**2

        if distance <= 1:
            points_inside_circle += 1

    pi = 4 * (points_inside_circle / total_points)
    return pi

# 设置随机点的数量
num_points = 10000000
approx_pi = calculate_pi(num_points)

print(f"Approximation of pi: {approx_pi}")
print(f"Error: {math.pi - approx_pi}")