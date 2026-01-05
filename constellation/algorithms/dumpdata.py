import csv
import json

with open(
    '/data/hhl/Constellation/xyh/Constellation/work_dirs/exp/visualization.json',
    'r'
) as f:
    data = json.load(f)

time_data = data['timeData']

max_sats = max(len(entry['constellation']) for entry in time_data)
max_tasks = max(len(entry['tasks']) for entry in time_data)

headers = []
headers.extend([f'earthRotation_{i}' for i in range(9)])

for sat_id in range(max_sats):
    prefix = f'satellite{sat_id}_'
    headers.append(prefix + 'id')
    headers.extend([
        prefix + 'eciLocation_x', prefix + 'eciLocation_y',
        prefix + 'eciLocation_z'
    ])
    headers.extend([
        prefix + 'eciVelocity_x', prefix + 'eciVelocity_y',
        prefix + 'eciVelocity_z'
    ])
    headers.extend([
        prefix + 'mrpAttitude_x', prefix + 'mrpAttitude_y',
        prefix + 'mrpAttitude_z'
    ])

# 动态列
for task_id in range(max_tasks):
    prefix = f'task{task_id}_'
    headers.extend([prefix + 'taskId', prefix + 'sensorType'])
    headers.extend([
        prefix + 'ecefLocation_x', prefix + 'ecefLocation_y',
        prefix + 'ecefLocation_z'
    ])

rows = []
for entry in time_data:
    row = []

    row.extend(entry['earthRotation'])

    satellites = entry['constellation']
    for sat in satellites:
        row.append(sat['satelliteId'])
        row.extend(sat['eciLocation'])
        row.extend(sat['eciVelocity'])
        row.extend(sat['mrpAttitude'])

    for _ in range(2 - len(satellites)):
        row.extend([None] * (1 + 3 + 3 + 3))  # id + 3坐标 + 3速度 + 3姿态

    tasks = entry['tasks']
    for task in tasks:
        row.append(task['taskId'])
        row.append(task['sensorType'])
        row.extend(task['ecefLocation'])

    for _ in range(max_tasks - len(tasks)):
        row.extend([None] * (2 + 3))  # taskId + sensorType + 3坐标

    rows.append(row)

with open('output.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    writer.writerows(rows)
