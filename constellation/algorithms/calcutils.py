import numpy as np


def include_angle(a, b):
    dot_product = np.dot(a, b)

    m_a = np.linalg.norm(a)
    m_b = np.linalg.norm(b)

    cos_value = dot_product / (m_a * m_b)

    return np.arccos(cos_value)


def deg2rad(degrees):
    return degrees * np.pi / 180.0


def compute_eci(e, a, i_deg, raan_deg, arg_perigee_deg, true_anomaly_deg):
    i = deg2rad(i_deg)
    raan = deg2rad(raan_deg)
    arg_perigee = deg2rad(arg_perigee_deg)
    true_anomaly = deg2rad(true_anomaly_deg)

    r = a * (1 - e**2) / (1 + e * np.cos(true_anomaly))

    x_orb = r * np.cos(true_anomaly)
    y_orb = r * np.sin(true_anomaly)
    z_orb = 0
    r_orb = np.array([x_orb, y_orb, z_orb])

    r_z_omega = np.array([[np.cos(arg_perigee), -np.sin(arg_perigee), 0],
                          [np.sin(arg_perigee),
                           np.cos(arg_perigee), 0], [0, 0, 1]])

    r_x_i = np.array([[1, 0, 0], [0, np.cos(i), -np.sin(i)],
                      [0, np.sin(i), np.cos(i)]])

    r_z_raan = np.array([[np.cos(raan), -np.sin(raan), 0],
                         [np.sin(raan), np.cos(raan), 0], [0, 0, 1]])

    r = r_z_raan @ r_x_i @ r_z_omega

    r_eci = r @ r_orb

    return r_eci


def geodetic2ecef(lat_deg, lon_deg, h):
    """将地理坐标（纬度、经度、高度）转换为 ECEF 坐标系。

    - lat_deg: 纬度，单位为度（°）
    - lon_deg: 经度，单位为度（°）
    - h: 高度 单位为米
    返回：
    - X, Y, Z: ECEF 坐标 单位为米
    """
    # WGS84
    a = 6378137.0  # 长半轴
    f = 1 / 298.257223563  # 扁率
    e_sq = f * (2 - f)  # 第一偏心率平方

    # 将角度转换为弧度
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)

    # 计算曲率半径
    n = a / np.sqrt(1 - e_sq * np.sin(lat)**2)

    # 计算 ECEF 坐标
    x = (n + h) * np.cos(lat) * np.cos(lon)
    y = (n + h) * np.cos(lat) * np.sin(lon)
    z = (n * (1 - e_sq) + h) * np.sin(lat)

    return x, y, z


def mrp_to_dcm(mrp):
    # 将MRP转换为方向余弦矩阵DCM。
    s = np.array(mrp)
    s_norm_sq = np.dot(s, s)
    I = np.eye(3)
    S = np.array([[0, -s[2], s[1]], [s[2], 0, -s[0]], [-s[1], s[0], 0]])
    # 根据 MRP 转换公式计算 DCM
    C = ((1 - s_norm_sq) * I + 2 * np.outer(s, s) - 2 * S) / (1 + s_norm_sq)**2
    return C


def compute_max_angular_velocity(constellation):

    satellites = constellation['satellites']
    orbits = {orbit['id_']: orbit for orbit in constellation['orbits']}

    max_angular_velocities = {}

    for sat in satellites:
        sat_id = sat['id']
        inertia = sat['inertia']
        inertia_matrix = np.array(inertia).reshape((3, 3))

        mrp = sat['mrp_attitude_bn']
        C = mrp_to_dcm(mrp)  # mrp -> dcm

        # 定义惯性坐标系中的垂直方向
        perp_dir_inertial = np.array([0, 0, 1])

        # interial -> body
        perp_dir_body = C.T @ perp_dir_inertial
        perp_dir_body /= np.linalg.norm(perp_dir_body)

        reaction_wheels = sat['reaction_wheels']
        L_max = 0.0  # angular momentum
        for rw in reaction_wheels:
            rw_dir = np.array(rw['rw_direction'])
            rw_max_momentum = rw['max_momentum']
            projection = np.dot(rw_dir, perp_dir_body)
            L_contribution = rw_max_momentum * abs(projection)
            L_max += L_contribution

        I_eff = perp_dir_body.T @ inertia_matrix @ perp_dir_body  # 转到垂直

        omega_max = L_max / I_eff  # in rad/s
        max_angular_velocities[sat_id] = omega_max

    return max_angular_velocities


if __name__ == "__main__":
    constellation_dict = {
        'satellites': [
            {
                'id': 0,
                'inertia': [900.0, 0.0, 0.0, 0.0, 800.0, 0.0, 0.0, 0.0, 600.0],
                'mass': 750.0,
                'center_of_mass': [0.0, 0.0, 0.0],
                'orbit': 0,
                'solar_panel': {
                    'direction': [1.0, 0.0, 0.0],
                    'area': 1.0,
                    'efficiency': 0.2
                },
                'sensor': {
                    'type': 1,
                    'enabled': 1,
                    'half_field_of_view': 5.0,
                    'power': 5.0
                },
                'battery': {
                    'capacity': 18000.0,
                    'percentage': 0.695491816117192
                },
                'reaction_wheels': [{
                    'rw_type': 'Honeywell_HR16',
                    'rw_direction': [1.0, 0.0, 0.0],
                    'max_momentum': 50.0,
                    'rw_speed_init': -462.8604640017897,
                    'power': 5.0,
                    'efficiency': 0.5
                }, {
                    'rw_type': 'Honeywell_HR16',
                    'rw_direction': [0.0, 1.0, 0.0],
                    'max_momentum': 50.0,
                    'rw_speed_init': -569.4989996941362,
                    'power': 5.0,
                    'efficiency': 0.5
                }, {
                    'rw_type': 'Honeywell_HR16',
                    'rw_direction': [0.0, 0.0, 1.0],
                    'max_momentum': 50.0,
                    'rw_speed_init': 688.0923696714565,
                    'power': 5.0,
                    'efficiency': 0.5
                }],
                'mrp_control': {
                    'k': 5.5,
                    'ki': -1.0,
                    'p': 30.0,
                    'integral_limit': -0.2
                },
                'true_anomaly': 136.9781740870415,
                'mrp_attitude_bn': [
                    0.08241958119604338, -0.6433596526211803,
                    -0.5239338711882789
                ]
            },
        ],
        'orbits': [
            {
                'id_': 0,
                'eccentricity': 0.010000111456346855,
                'semi_major_axis': 7000000.390845305,
                'inclination': 3.078324659199975e-06,
                'right_ascension_of_the_ascending_node': 88.54914134922846,
                'argument_of_perigee': 91.45126790352177
            },
            {
                'id_': 1,
                'eccentricity': 0.010000075325443636,
                'semi_major_axis': 7000000.510378544,
                'inclination': 60.00000032744095,
                'right_ascension_of_the_ascending_node': 90.00000011247428,
                'argument_of_perigee': 90.00069633064284
            },
        ]
    }

    max_angular_vel = compute_max_angular_velocity(constellation_dict)

    for sat_id, omega in max_angular_vel.items():
        print(f"sat {sat_id} omega_max: {omega:.6f} rad/s")
