"""This module is used to generate data related to satellites and tasks.

And we use secrets rather than random.

In particular, secrets should be used in preference to the default
pseudo-random number generator in the random module, which is designed
for modelling and simulation, not security or cryptography.
"""
import sys
import json
import random
import secrets
from typing import Union

TOTAL_SENSOR_TYPES = [1, 2]
EARTH_RADIUS = 6371  # 地球半径（单位：公里）

# 用于经验公式
m_momentum = 0
max_in = 0

TOTAL_SATELLITES_TYPES = {
    "small1": {
        "mass": (8, 16),
        "reaction_wheels": {
            "max_momentum": (300, 700),  # 1e-3
            "rw_speed_init": (300, 900),
            "power": (1.35, 3),
            "efficiency": (0.3, 0.7),
        },
    },
    "small2": {
        "mass": (10, 20),
        "reaction_wheels": {
            "max_momentum": (10, 476),  # 1e-3
            "rw_speed_init": (500, 700),
            "power": (2.5, 5.1),
            "efficiency": (0.3, 0.7),
        },
    },
    "medium1": {
        "mass": (30, 80),
        "reaction_wheels": {
            "max_momentum": (100, 500),  # 1e-3
            "rw_speed_init": (100, 500),
            "power": (2, 6),
            "efficiency": (0.3, 0.7),
        },
    },
    "medium": {
        "mass": (30, 100),
        "reaction_wheels": {
            "max_momentum": (800, 1200),  # 1e-3
            "rw_speed_init": (400, 800),
            "power": (4, 7),
            "efficiency": (0.3, 0.7),
        },
    },
    "large": {
        "mass": (80, 150),
        "reaction_wheels": {
            "max_momentum": (1800, 2200),  # 1e-3
            "rw_speed_init": (400, 800),
            "power": (6, 9),
            "efficiency": (0.3, 0.7),
        },
    },
}


def secrets_random_round(a, b, c):
    return round(a + (secrets.SystemRandom().random() * (b - a)), c)


def secrets_random_int(a, b):
    return a + secrets.randbelow(b - a)


# 生成指定数量个 orbits
def generate_orbits(
    num_orbits_: int,
    eccentricity_: list[float],
    semi_major_axis_: list[int],
    inclination_: list[float],
    right_ascension_of_the_ascending_node_: list[float],
    argument_of_perigee_: list[float],
    is_random: bool = True,
) -> list[dict[str, Union[int, float]]]:
    """Introduce the parameters.

    Parameters:
    num_orbits(int) : 轨道数量
    is_random(bool) : 决定是否随机生成数据，如果为 False 要确保输入的数据均不为空
    eccentricity(list[float]): 离心率
    semi_major_axis(list[int]) : 半长轴
    inclination(list[float]) : 倾角
    right_ascension_of_the_ascending_node(list[float]) : 升交点经度
    argument_of_perigee(list[float]) : 近地点幅角

    如果传入 list 参数，则要求所有 list 长度等于 num_orbits
    """
    orbits_ = []

    for id_ in range(num_orbits_):
        if is_random:
            e_ = secrets_random_round(0.01, 0.02, 3)
            s_ = (
                secrets.randbelow(1800) + 200 + EARTH_RADIUS
            ) * 1000  # LEO 离地距离通常在 200~2000km 之间
            i_ = secrets_random_round(0, 180, 3)
            r_ = secrets_random_round(-180, 180, 3)
            a_ = secrets_random_round(0, 360, 3)
        else:
            e_ = eccentricity_[id_]
            s_ = semi_major_axis_[id_]
            i_ = inclination_[id_]
            r_ = right_ascension_of_the_ascending_node_[id_]
            a_ = argument_of_perigee_[id_]

        orbit_ = {
            "id": id_,
            "eccentricity": e_,  # 轨道离心率，描述轨道形状的参数。0 表示圆形轨道，1 表示高度离心轨道，单位 无
            "semi_major_axis": s_,  # 半长轴，是椭圆轨道的两个半轴之一，表示轨道椭圆的大小，单位 m
            "inclination": i_,  # 倾角，是轨道平面与参考平面（通常是地球赤道平面）之间的夹角，单位 °
            "right_ascension_of_the_ascending_node": r_,  # 升交点经度，单位 °
            "argument_of_perigee": a_,  # 近地点幅角，描述轨道上近地点与升交点之间的角度，单位 °
        }
        orbits_.append(orbit_)

    return orbits_


# 生成多个 satellites
# 根据轨道的 id 生成几个卫星均匀分布的例子，假设每个轨道上卫星数量为 1 ~ 6
MAX_SATELLITES_NUM = 6
MIN_SATELLITES_NUM = 1


def generate_satellites(
    num_orbits_: int,
    satellite_types_: list,
    inertia_: list[list],
    mass_: list[float],
    center_of_mass_: list[list],
    orbit_ids_: list[int],
    solar_panel_: dict,
    sensor_: dict,
    battery_: dict,
    reaction_wheels_: list[list],
    mrp_control_: dict,
    true_anomalys_: list[float],
    mrp_attitude_bn_: list[list],
) -> tuple[list[dict[str, Union[int, float, dict]]], int]:
    """Introduce the parameters.

    Parameters:
    num_orbits_(int) : 轨道数量
    satellite_types_(list) : 卫星类型
    inertia_(list[list]) : 卫星转动惯量张量
    mass_(list[float]) : 卫星质量
    center_of_mass_(list[list]) : 卫星质心
    orbit_ids_(list[int]) : 每颗卫星对应的轨道 id
    solar_panel_(dict) : 太阳能板，每个 key 对应的 value(list)
    sensor_(dict) : 感应器，每个 key 对应的 value(list)
    battery_(dict) : 电池，每个 key 对应的 value(list)
    reaction_wheels_(list[list]) : 反应轮参数
    mrp_control_(dict) : MRP 控制参数
    true_anomalys_(list) : 真近角
    mrp_attitude_bn_(list[list]) : 初始姿态

    参数 orbit_ids 与 true_anomalys 必须同为 None 或同不为 None
    其余参数如果传入不为 None 需要确保 list 长度一致
    """
    satellites_ = []
    total_satellites_num = 0

    if not orbit_ids_ or not true_anomalys_:
        orbit_ids_, true_anomalys_ = generate_orbit_and_true_anomalys(
            num_orbits_,
        )
    elif len(orbit_ids_) != len(true_anomalys_):
        print(
            "Error: Parameters orbits ids and true anomalys must ",
            "have the same state! Please check.",
        )
        return [], 0

    total_satellites_num = len(orbit_ids_)

    if len(satellite_types_) == 0:
        satellite_types_ = generate_satellite_types_params(
            total_satellites_num,
        )

    for satellite_id_, (orbit_id, true_anomaly) in enumerate(
        zip(orbit_ids_, true_anomalys_),
    ):
        satellite_type_ = satellite_types_[satellite_id_]
        mass_range = TOTAL_SATELLITES_TYPES[satellite_type_]["mass"]
        reaction_wheels_round = TOTAL_SATELLITES_TYPES[satellite_type_][
            "reaction_wheels"]

        inertia_params = generate_inertia_params(inertia_, satellite_id_)
        mass_params = generate_mass_params(mass_, satellite_id_, mass_range)
        center_of_mass_params = generate_center_of_mass_params(
            center_of_mass_,
            satellite_id_,
        )
        solar_panel_params = generate_solar_panel_params(
            solar_panel_,
            satellite_id_,
        )
        sensor_params = generate_sensor_params(sensor_, satellite_id_)
        battery_params = generate_battery_params(battery_, satellite_id_)
        reaction_wheels_params = generate_reaction_wheels_params(
            reaction_wheels_,
            satellite_id_,
            reaction_wheels_round,
        )
        mrp_control_params = generate_mrp_control_params(
            mrp_control_,
            satellite_id_,
        )
        mrp_attitude_bn_params = generate_mrp_attitude_bn_params(
            mrp_attitude_bn_,
            satellite_id_,
        )

        satellite = {
            "id": satellite_id_,
            "inertia": inertia_params,  # 卫星转动惯量张量
            "mass": mass_params,  # 卫星质量
            "center_of_mass": center_of_mass_params,  # 卫星质心位置
            "orbit": orbit_id,  # 卫星所处的轨道的 id
            "solar_panel": solar_panel_params,
            "sensor": sensor_params,
            "battery": battery_params,
            "reaction_wheels": reaction_wheels_params,
            "mrp_control": mrp_control_params,
            "true_anomaly": true_anomaly,  # 真近点角，描述卫星在轨道上的位置，单位 °
            "mrp_attitude_bn": mrp_attitude_bn_params,
        }
        satellites_.append(satellite)

    return satellites_, total_satellites_num


def generate_satellite_types_params(satellites_num):
    satellite_types_params = [
        secrets.choice(list(TOTAL_SATELLITES_TYPES.keys()))
        for _ in range(satellites_num)
    ]
    return satellite_types_params


def generate_inertia_params(inertia_, i):
    global max_in
    if not inertia_:
        inertia_params = [random.uniform(50,150), 0, 0,
                          0, random.uniform(50,150), 0,
                          0, 0, random.uniform(50,150)]
    else:
        inertia_params = inertia_[i]
    for i in inertia_params:
        if i > max_in:
            max_in = i
    return inertia_params


def generate_mass_params(mass_, i, mass_range):
    if not mass_:
        mass_params = secrets_random_round(mass_range[0], mass_range[1], 3)
    else:
        mass_params = mass_[i]

    return mass_params


def generate_center_of_mass_params(center_of_mass_, i):
    if not center_of_mass_ == 0:
        center_of_mass_params = [0.0, 0.0, 0.0]
    else:
        center_of_mass_params = center_of_mass_[i]

    return center_of_mass_params


def generate_orbit_and_true_anomalys(num_orbits_):
    orbit_ids_ = []
    true_anomalys_ = []

    for orbit_id_ in range(num_orbits_):
        satellites_num = secrets_random_int(
            MIN_SATELLITES_NUM,
            MAX_SATELLITES_NUM,
        )
        orbit_ids_.extend([orbit_id_] * satellites_num)

        true_anomaly_init = round(secrets.randbelow(360 * 1e3) / 1e3)
        phase_difference = 360 / satellites_num
        for i in range(satellites_num):
            tmp = true_anomaly_init + phase_difference * i
            if tmp > 360:
                tmp -= 360
            true_anomalys_.append(tmp)

    return orbit_ids_, true_anomalys_


def generate_solar_panel_params(solar_panel_, i):
    if not solar_panel_:
        direction_ = [
            secrets_random_round(-1, 1, 3),
            secrets_random_round(-1, 1, 3),
            secrets_random_round(-1, 1, 3),
        ]  # 生成朝向参数
        area_ = secrets_random_round(1, 10, 3)  # 生成面积参数（1 ~ 10 m^2）
        efficiency_ = secrets_random_round(0, 1, 3)  # 生成效率参数（0 ~ 1）
    else:
        direction_ = solar_panel_["direction"][i]
        area_ = solar_panel_["area"][i]
        efficiency_ = solar_panel_["efficiency"][i]

    return {"direction": direction_, "area": area_, "efficiency": efficiency_}


def generate_sensor_params(sensor_, i):
    if not sensor_:
        sensor_type_ = secrets.choice(TOTAL_SENSOR_TYPES)  # 生成传感器类型
        enabled_ = secrets.choice([True, False])  # 生成是否启用传感器
        half_field_of_view_ = round(
            secrets_random_round(4, 6, 3),
            2,
        )  # 生成半视场角（4 ~ 6 °）
        power_ = secrets_random_round(1, 10, 3)  # 生成功率参数（1 ~ 10 w）
    else:
        sensor_type_ = sensor_["sensor_type"][i]
        enabled_ = sensor_["enabled"][i]
        half_field_of_view_ = sensor_["half_field_of_view"][i]
        power_ = sensor_["power"][i]

    return {
        "type": sensor_type_,
        "enabled": enabled_,
        "half_field_of_view": half_field_of_view_,
        "power": power_,
    }


def generate_battery_params(battery_, i):
    if not battery_:
        capacity_ = secrets_random_int(5, 50) * 1e3  # 生成电池容量
        percentage_ = secrets_random_round(0, 1, 3)  # 生成充电状态
    else:
        capacity_ = battery_["capacity"][i]
        percentage_ = battery_["percentage"][i]

    return {"capacity": capacity_, "percentage": percentage_}


def generate_reaction_wheels_params(
    reaction_wheels_,
    i,
    reaction_wheels_range,
):
    rw_type = random.choice(["12","14","16"])
    max_momentum_range = reaction_wheels_range["max_momentum"]
    rw_speed_init_range = reaction_wheels_range["rw_speed_init"]
    power_range = reaction_wheels_range["power"]
    efficiency_range = reaction_wheels_range["efficiency"]

    if rw_type == "12":
        max_momentum_ = random.choice([12.0,25.0,50.0])
    elif rw_type == "14":
        max_momentum_ = random.choice([75.0,25.0,50.0])
    else:
        max_momentum_ = random.choice([75.0,100.0,50.0])


    rw_speed_init_ = secrets_random_round(
        rw_speed_init_range[0],
        rw_speed_init_range[1],
        3,
    )
    power_ = secrets_random_round(
        power_range[0],
        power_range[1],
        3,
    )
    efficiency_ = secrets_random_round(
        efficiency_range[0],
        efficiency_range[1],
        3,
    )
    if not reaction_wheels_:
        reaction_wheels_params = [{
            "rw_type": "Honeywell_HR" + rw_type,
            "rw_direction": [1.0, 0.0, 0.0],
            "max_momentum": max_momentum_,
            "rw_speed_init": rw_speed_init_,
            "power": power_,
            "efficiency": efficiency_,
        }, {
            "rw_type": "Honeywell_HR" + rw_type,
            "rw_direction": [0.0, 1.0, 0.0],
            "max_momentum": max_momentum_,
            "rw_speed_init": rw_speed_init_,
            "power": power_,
            "efficiency": efficiency_,
        }, {
            "rw_type": "Honeywell_HR" + rw_type,
            "rw_direction": [0.0, 0.0, 1.0],
            "max_momentum": max_momentum_,
            "rw_speed_init": rw_speed_init_,
            "power": power_,
            "efficiency": efficiency_,
        }]
    else:
        reaction_wheels_params = reaction_wheels_[i]
    global m_momentum
    m_momentum = max_momentum_
    return reaction_wheels_params


def generate_mrp_control_params(mrp_control_, i):
    if not mrp_control_:
        k = 1.05 * (max_in / m_momentum)
        ki = 0.001
        p = 4.1 * k
        integral_limit = 0.01
    else:
        k = mrp_control_["k"][i]
        ki = mrp_control_["ki"][i]
        p = mrp_control_["p"][i]
        integral_limit = mrp_control_["integral_limit"][i]

    return {"k": k, "ki": ki, "p": p, "integral_limit": integral_limit}


def generate_mrp_attitude_bn_params(mrp_attitude_bn_, i):
    if not mrp_attitude_bn_:
        mrp_attitude_bn_params = [0.0, 0.0, 0.0]
    else:
        mrp_attitude_bn_params = mrp_attitude_bn_[i]

    return mrp_attitude_bn_params


# 生成多个 tasks
# 还需要仅 id 和 sensor type 不一样的 tasks
def generate_tasks(
    release_times_: list[int],
    due_times_: list[int],
    durations_: list[int],
    coordinate_: list[tuple[float, float]],
    sensor_types_: list[int],
    is_random: bool = True,
) -> tuple[list[dict[str, Union[int, float, dict]]], int]:
    tasks_data_ = []
    num_tasks_ = 0

    if is_random:
        is_same = [True, False]
        len(TOTAL_SENSOR_TYPES)
        task_id_ = -1

        for _, (x_, y_) in enumerate(coordinate_):
            flag = secrets.choice(
                is_same,
            )  # 以 0.5 的概率决定是否生成仅 id 和 sensor type 不同的 tasks

            # time step
            release_time_ = secrets_random_int(
                0,
                900,
            )  # release_time < due_time
            due_time_ = secrets_random_int(
                release_time_ + 20,
                1000,
            )  # due_time
            duration_ = secrets_random_int(1, 5)  # duration 取值不能太大

            # sensor type
            task_num = 2 if flag else 1  # 确定生成 "仅 id 和 sensor type 不同" 的task数量
            sensor_types_ = random.sample(TOTAL_SENSOR_TYPES, task_num)

            for sensor_type_ in sensor_types_:
                task_id_ += 1
                tasks_data_.append({
                    "id": task_id_,
                    "release_time": release_time_,  # 数据的开始时间
                    "due_time": due_time_,  # 数据的截止时间
                    "duration": duration_,  # 数据的持续时间
                    "coordinate": {  # 坐标
                        "x": x_,  # 经度
                        "y": y_,  # 纬度
                    },
                    "sensor_type": sensor_type_,  # 传感器类型
                })
        num_tasks_ = task_id_ + 1
    else:
        for task_id_, (x_, y_) in enumerate(coordinate_):
            release_time_ = release_times_[task_id_]
            due_time_ = due_times_[task_id_]
            duration_ = durations_[task_id_]
            sensor_type_ = sensor_types_[task_id_]

            tasks_data_.append({
                "id": task_id_,
                "release_time": release_time_,  # 数据的开始时间
                "due_time": due_time_,  # 数据的截止时间
                "duration": duration_,  # 数据的持续时间
                "coordinate": {  # 坐标
                    "x": x_,  # 经度
                    "y": y_,  # 纬度
                },
                "sensor_type": sensor_type_,  # 传感器类型
            })
        num_tasks_ = len(coordinate_)

    return tasks_data_, num_tasks_


if __name__ == '__main__':
    # 构建 constellations 数据
    # orbits 数据
    NUM_ORBITS = int(sys.argv[1])  # 指定要生成的轨道数量
    # eccentricity = [0.01] * NUM_ORBITS
    eccentricity = [random.uniform(0.0, 0.002) for _ in range(NUM_ORBITS)]
    # semi_major_axis = [7000] * NUM_ORBITS
    semi_major_axis = [random.uniform(6500.0, 8000.0) * 1000 for _ in range(NUM_ORBITS)] # LEO orbits
    # inclination = [0., 60.]
    inclination = [random.uniform(0.0, 180.0) for _ in range(NUM_ORBITS)]
    # r = [90.] * NUM_ORBITS
    r = [random.uniform(0.0, 360.0) for _ in range(NUM_ORBITS)]
    # argument_of_perigee = [90.] * NUM_ORBITS
    argument_of_perigee = [random.uniform(0.0, 360.0) for _ in range(NUM_ORBITS)]

    orbits = generate_orbits(
        NUM_ORBITS,
        is_random=False,
        eccentricity_=eccentricity,
        semi_major_axis_=semi_major_axis,
        inclination_=inclination,
        right_ascension_of_the_ascending_node_=r,
        argument_of_perigee_=argument_of_perigee,
    )

    # satellites 数据
    orbit_ids = [i for i in range(NUM_ORBITS)]
    solar_panel: dict = {"direction": [], "area": [], "effeicency": []}
    sensor: dict = {
        "type": [],
        "enabled": [],
        "half_field_of_view": [],
        "power": [],
    }
    battery: dict = {"capacity": [], "percentage": []}
    true_anomalys = [random.uniform(-180.0, 180.0) for _ in range(NUM_ORBITS)]

    satellites, num_satellites = generate_satellites(
        NUM_ORBITS,
        satellite_types_=[],
        inertia_=[],
        mass_=[],
        center_of_mass_=[],
        orbit_ids_=orbit_ids,
        solar_panel_={},
        sensor_={},
        battery_={},
        reaction_wheels_=[],
        mrp_attitude_bn_=[],
        true_anomalys_=true_anomalys,
        mrp_control_={},
    )

    constellation_data = {"orbits": orbits, "satellites": satellites}

    CONSTELLATION_PATH = f'./data/constellations/{sys.argv[2]}.json'
    with open(CONSTELLATION_PATH, 'w') as outfile:
        json.dump(constellation_data, outfile, indent=4)

    print(
        f"Data with {NUM_ORBITS} orbits and {num_satellites} satellites",
        f"has been successfully generated and saved to {CONSTELLATION_PATH} file.",
    )

    # # 构建 tasks数据
    # x = [4., -2., -1., -20., -30., -40.]
    # y = [180., 175., 168., 176., 180., 180.]

    # release_times: list = []
    # due_times: list = []
    # durations: list = []
    # coordinate = list(zip(x, y))
    # sensor_types: list = []

    # tasks_data, num_tasks = generate_tasks(
    #     is_random=True,
    #     release_times_=release_times,
    #     due_times_=due_times,
    #     durations_=durations,
    #     coordinate_=coordinate,
    #     sensor_types_=sensor_types,
    # )

    # TASK_PATH = '../tlxd_workspace/data/syn_data/tasks/0.json'
    # with open(TASK_PATH, 'w') as outfile:
    #     json.dump(tasks_data, outfile, indent=4)

    # print(
    #     f"{num_tasks} sets of data have been successfully ",
    #     f"generated and saved to {TASK_PATH} file.",
    # )
