import json
import os
import random
from pathlib import Path
import random
from tqdm import tqdm

def generate_orbit(orbit_id):
    """生成随机轨道参数"""
    return {
        "id": orbit_id,
        "eccentricity": round(random.uniform(0, 0.005), 6),
        "semi_major_axis": round(random.uniform(6.8e6, 8e6), 1),
        "inclination": round(random.uniform(0, 180), 1),
        "right_ascension_of_the_ascending_node": round(random.uniform(0, 360), 1),
        "argument_of_perigee": round(random.uniform(0, 360), 1)
    }

def process_files(input_dir, sample_count):
    constellation = []
    orbits = []
    all_sats = []
    
    # 收集所有卫星数据
    for json_file in Path(input_dir).glob("*.json"):
        with open(json_file, 'r') as f:
            try:
                data = json.load(f)
                all_sats.extend(data.get("satellites", []))
            except Exception as e:
                print(f"跳过损坏文件 {json_file}: {str(e)}")
    
    # 随机抽样（默认抽取50%）
    selected_sats = random.sample(all_sats, sample_count)
    
    # 生成递增id
    for idx, sat in enumerate(selected_sats):
        # 保留原始参数，仅修改orbit
        new_entry = {
            "id": idx,
            "orbit": idx,  # 强制orbit与id一致
            "mass": sat.get("mass"),
            "center_of_mass": sat.get("center_of_mass"),
            "inertia": sat.get("inertia"),
            "solar_panel": sat.get("solar_panel"),
            "sensor": sat.get("sensor"),
            "battery": {'capacity': random.uniform(8000,30000), 'percentage': random.uniform(0.1,1.0)},
            "reaction_wheels": sat.get("reaction_wheels"),
            "mrp_control": sat.get("mrp_control"),
            "mrp_attitude_bn": sat.get("mrp_attitude_bn", [0,0,0]),
            "true_anomaly": random.uniform(0,360)
        }
        constellation.append(new_entry)
        orbits.append(generate_orbit(idx))
    
    return {
        "satellites": constellation,
        "orbits": orbits
    }

def main():
    input_dir = "useful_sats"
    output_dir = "../../data/constellations/"

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))


    for idx in range(rank, 1040000, world_size):
        # 调整抽样比例
        combined_data = process_files(input_dir, sample_count=random.randint(1, 50))  
        print(f"running:{idx}")
        with open(output_dir + f"{idx}.json", 'w') as f:
            json.dump(combined_data, f, indent=2, ensure_ascii=False)

    # for p in cmds:
    #     p.close()

if __name__ == "__main__":
    main()