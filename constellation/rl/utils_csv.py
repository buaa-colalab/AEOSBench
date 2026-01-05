import argparse
import pathlib
from statistics import mean
import pandas as pd
import torch

from constellation.new_transformers.dataset import Annotation

# def pth_to_csv(pth_path) -> None:
#     completion_rate_pth = torch.load(pth_path, weights_only=False)
#     work_dir = pathlib.Path('work_dirs') / f'rl_eval_rl_loaded'
#     work_dir.mkdir(parents=True, exist_ok=True)
#     with open(work_dir / 'completion_rates.csv', 'w') as f:
#         for (constellation_id,
#              tasks_id), completion_rate in completion_rate_pth.items():
#             f.write(
#                 f"{constellation_id},{tasks_id},{completion_rate}\n",
#             )
#             f.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='csv_combine')
    parser.add_argument('output', type=pathlib.Path)
    parser.add_argument('csv', type=pathlib.Path, nargs='+')
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()

    combined_completion_rates: dict[tuple, float] | None = dict()
    for csv_path in args.csv:
        if not csv_path.exists():
            raise FileNotFoundError(f"File {csv_path} does not exist.")
        df = pd.read_csv(
            csv_path,
            names=['constellation_id', 'tasks_id', 'completion_rate'],
        )
        df = df.set_index(['constellation_id', 'tasks_id'])
        completion_rates = df.to_dict()['completion_rate']

        for key, value in completion_rates.items():
            combined_completion_rates[key] = max(
                combined_completion_rates.get(key, 0.0), value
            )

    # Write the combined results to a CSV file
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        for (constellation_id,
             tasks_id), completion_rate in combined_completion_rates.items():
            f.write(f"{constellation_id},{tasks_id},{completion_rate}\n")

    print(
        f"Mean combined_completion_rates: {mean(combined_completion_rates.values())}"
    )
    filtered_values = [
        value for value in combined_completion_rates.values() if value >= 0.05
    ]
    if filtered_values:
        print(
            f"Mean combined_completion_rates "
            f"(filtered >= 0.05, {len(filtered_values)}/{len(combined_completion_rates)}): "
            f"{mean(filtered_values)}"
        )
    else:
        print("No values >= 0.05 to calculate mean.")


if __name__ == '__main__':
    main()
