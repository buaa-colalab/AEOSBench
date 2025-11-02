import json
import sys
import random
import numpy as np
from tqdm import tqdm


def generate_satellite(satellite_id):
    # 生成惯性矩阵（对角矩阵）
    inertia = [
        round(random.uniform(50, 200), 6), 0.0, 0.0, 0.0,
        round(random.uniform(50, 200), 6), 0.0, 0.0, 0.0,
        round(random.uniform(50, 200), 6)
    ]

    # 卫星质量
    mass = round(random.uniform(50, 200), 3)

    # 太阳能板方向（归一化）
    while True:
        direction = [random.uniform(-1, 1) for _ in range(3)]
        norm = np.linalg.norm(direction)
        if norm > 1e-6:  # 避免零向量
            direction = [round(d / norm, 3) for d in direction]
            break

    solar_panel = {
        "direction": direction,
        "area": round(random.uniform(5, 10), 3),
        "efficiency": round(random.uniform(0.1, 0.6), 3)
    }

    # 传感器参数
    sensor = {
        "type": 1,
        "enabled": random.choice([True, False]),
        "half_field_of_view": round(random.uniform(0.5, 1.5), 2),  # TODO
        "power": round(random.uniform(2, 8), 3)
    }

    # 电池参数
    battery = {
        "capacity": round(random.uniform(8000, 30000), 1),
        "percentage": round(random.uniform(0.7, 1.0), 3)  # TODO: 0-1
    }

    # 反作用轮（正交三轴）
    rw_type = random.choice(["12", "14", "16"])
    if rw_type == "12":
        max_momentum = random.choice([12.0, 25.0, 50.0])
    elif rw_type == "14":
        max_momentum = random.choice([75.0, 25.0, 50.0])
    elif rw_type == "16":
        max_momentum = random.choice([100.0, 75.0, 50.0])
    reaction_wheels = []
    for axis in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]:
        reaction_wheel = {
            "rw_type": "Honeywell_HR" + rw_type,
            "rw_direction": axis,
            "max_momentum": max_momentum,
            "rw_speed_init": round(random.uniform(400, 750), 3),
            "power": round(random.uniform(5, 7), 3),
            "efficiency": round(random.uniform(0.5, 0.6), 3)
        }
        reaction_wheels.append(reaction_wheel)

    # MRP控制参数
    mrp_k = random.uniform(0, 10)
    mrp_control = {
        "k": round(mrp_k, 6),
        "ki": round(random.uniform(0, 0.001), 6),
        "p": round(random.uniform(2, 5) * mrp_k, 6),
        "integral_limit": round(random.uniform(0, 0.001), 6),
    }

    # 真近点角
    true_anomaly = 0.0

    return {
        "id": satellite_id,
        "inertia": inertia,
        "mass": mass,
        "center_of_mass": [0.0, 0.0, 0.0],
        "orbit": satellite_id,  # 保持原始文件中的orbit分配（0和1）
        "solar_panel": solar_panel,
        "sensor": sensor,
        "battery": battery,
        "reaction_wheels": reaction_wheels,
        "mrp_control": mrp_control,
        "true_anomaly": true_anomaly,
        "mrp_attitude_bn": [0.0, 0.0, 0.0]
    }


# 固定轨道参数
orbits = [{
    "id": 0,
    "eccentricity": 0.0001,
    "semi_major_axis": 7200000.0,
    "inclination": 0.0,
    "right_ascension_of_the_ascending_node": 0.0,
    "argument_of_perigee": 0.0
}]

for id in tqdm(range(1000)):
    satellites = [generate_satellite(0)]

    # 保存为JSON文件
    with open(f'./tools/mrp_fit/sats/{id}.json', 'w') as f:
        json.dump({"orbits": orbits, "satellites": satellites}, f, indent=4)

#####################tasks######################

# def generate_sensor_data(num_entries=36):
#     data = []
#     for idx in range(num_entries):
#         entry = {
#             "id": idx,
#             "release_time": 0,
#             "due_time": 7200,
#             "duration": 30,
#             "coordinate": {
#                 "y": -180.0 + idx * 10.0,
#                 "x": round(random.uniform(-10, 10), 6)
#             },
#             "sensor_type": 1
#         }
#         data.append(entry)
#     return data

# # 生成
# task_data = generate_sensor_data()

# # 保存为JSON文件
# with open(f'./toolkits/mrp_fit/tasks/0.json', 'w') as f:
#     json.dump(task_data, f, indent=4)
