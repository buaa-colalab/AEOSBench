import json
import random

import numpy as np


def kepler_to_cartesian(a, e, i, omega, w, M):
    mu = 3.986004418e14  # m^3/s^2

    i = np.radians(i)
    omega = np.radians(omega)
    w = np.radians(w)
    M = np.radians(M)
    E = M
    for _ in range(10):
        E = M + e * np.sin(E)
    nu = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))

    r = a * (1 - e * np.cos(E))
    x = r * np.cos(nu)
    y = r * np.sin(nu)

    R3_omega = np.array([[np.cos(omega), -np.sin(omega), 0],
                         [np.sin(omega), np.cos(omega), 0], [0, 0, 1]])

    R1_i = np.array([[1, 0, 0], [0, np.cos(i), -np.sin(i)],
                     [0, np.sin(i), np.cos(i)]])

    R3_w = np.array([[np.cos(w), -np.sin(w), 0], [np.sin(w),
                                                  np.cos(w), 0], [0, 0, 1]])

    r_orbit = np.array([x, y, 0])
    r_eci = R3_omega @ R1_i @ R3_w @ r_orbit

    return r_eci


def generate_ground_track(orbit_params, num_points=100):
    points = []
    mu = 3.986004418e14
    a = orbit_params['semi_major_axis']
    n = np.sqrt(mu / a**3)  # 平均运动，rad/s
    omega_earth = np.radians(360) / (
        23 * 3600 + 56 * 60 + 4.09053
    )  # 地球自转角速度，rad/s

    for M in np.linspace(0, 360, num_points):
        M_rad = np.radians(M)
        t = M_rad / n  # 时间（秒）
        earth_rotation = omega_earth * t

        pos = kepler_to_cartesian(
            a, orbit_params['eccentricity'], orbit_params['inclination'],
            orbit_params['right_ascension_of_the_ascending_node'],
            orbit_params['argument_of_perigee'], M
        )

        r = np.linalg.norm(pos)
        lat = np.arcsin(pos[2] / r)
        lon = np.arctan2(pos[1], pos[0]) - earth_rotation

        # 调整经度到[-π, π]范围
        lon = (lon + np.pi) % (2 * np.pi) - np.pi

        lat_deg = np.degrees(lat)
        lon_deg = np.degrees(lon)
        points.append((lon_deg, lat_deg))

    return points


def generate_tasks(constellation_file, num_tasks=10):

    with open(constellation_file, 'r') as f:
        constellation = json.load(f)

    orbits = constellation['orbits']
    tasks = []

    all_track_points = []
    for orbit in orbits:
        track_points = generate_ground_track(orbit, num_points=1000)
        all_track_points.extend(track_points)

    for i in range(num_tasks):
        base_point = random.choice(all_track_points)

        # 添加最多±1度的随机偏移
        lon_offset = random.uniform(-1, 1)
        lat_offset = random.uniform(-1, 1)

        release_time = random.randint(0, 1800)
        # release_time = 0
        due_time = min(3600 + random.randint(100, 4000), 7200)
        # due_time = 7200

        task = {
            "id": i,
            "release_time": release_time,
            "due_time": due_time,
            "duration": random.randint(1, 10),
            "coordinate": {
                "x": base_point[0] + lon_offset,  # 经度
                "y": base_point[1] + lat_offset  # 纬度
            },
            "sensor_type": 1
        }
        tasks.append(task)

    return tasks


if __name__ == "__main__":
    constellation_file = "./data/constellations/2.json"
    output_file = "./data/tasks/2.json"

    tasks = generate_tasks(constellation_file, num_tasks=50)

    # 保存生成的任务
    with open(output_file, 'w') as f:
        json.dump(tasks, f, indent=4)

    print(f"Generated {len(tasks)} tasks and saved to {output_file}")
