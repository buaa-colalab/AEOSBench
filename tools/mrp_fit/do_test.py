import json

from constellation.controller import test_mrp as simulate


def main():
    # parser = argparse.ArgumentParser('Tmp')
    # parser.add_argument('x_max', type=int)
    # args = parser.parse_args()

    combinations = list(range(1000))
    finish_list = []

    for idx in range(len(combinations)):
        x = combinations[idx]
        for _ in range(3):
            try:
                cr = simulate(x)
                if cr > 0.5:
                    finish_list.append((x, cr))
                    break
            except:
                print(f"Error in simulation for {x}")

    # for p in cmds:
    #     p.close()
    with open("tools/mrp_fix/result.txt", "w") as f:
        for i in finish_list:
            f.write(f"{i[0]},{i[1]}\n")


if __name__ == "__main__":
    main()
