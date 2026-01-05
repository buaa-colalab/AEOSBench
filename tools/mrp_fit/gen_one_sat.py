from constellation.data import Constellation
for idx in range(1000):
    constellation = Constellation.sample_mrp_fit_single(satellite_id=0)
    with open(f'data/mrp_sat/{idx}.json', 'w') as f:
        constellation.dump_std_json(f)

# --------------------- tasks ---------------------

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
