import json
import os

from constellation.controller import test_mrp as simulate


def main():
    # parser = argparse.ArgumentParser('Tmp')
    # parser.add_argument('x_max', type=int)
    # args = parser.parse_args()

    combinations = list(range(1000))
    finish_list = []

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    for idx in range(rank, len(combinations), world_size):
        x = combinations[idx]
        for _ in range(2):
            try:
                cr = simulate(x)
                # print(x, cr)
                if cr > 0.5:
                    finish_list.append((x, cr))
                    break
            except Exception as e:
                print(f"Error in simulation for {x}", e)

    # for p in cmds:
    #     p.close()
    with open(f"data/mrp_result/{rank}.txt", "w") as f:
        for i in finish_list:
            f.write(f"{i[0]},{i[1]}\n")


if __name__ == "__main__":
    main()
