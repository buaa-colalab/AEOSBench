import csv
import os
import shutil

# 定义路径
csv_path = 'data/mrp_result/'
sats_dir = 'data/mrp_sat/'
output_dir = 'tools/mrp_fit/useful_sats/'

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 读取CSV文件并提取第一列数字
numbers = []
for filename in os.listdir(csv_path):
    file_path = os.path.join(csv_path, filename)
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            print(row)
            if row and float(row[0]) > 0.99:  # 跳过空行
                numbers.append(row[0])
# 去重并转换为整数
numbers = list(map(int, set(numbers)))

# 复制文件
for id, num in enumerate(numbers):
    src_file = os.path.join(sats_dir, f"{num}.json")
    dest_file = os.path.join(output_dir, f"{id}.json")
    
    if os.path.exists(src_file):
        shutil.copy(src_file, dest_file)
        print(f"已复制: {src_file} -> {dest_file}")
    else:
        print(f"警告: 文件 {src_file} 不存在，已跳过")

print("操作完成！")