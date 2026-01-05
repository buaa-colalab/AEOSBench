import argparse
import pathlib

import pandas as pd
import todd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Merge CSVs')
    parser.add_argument('output', type=pathlib.Path)
    parser.add_argument('csvs', type=pathlib.Path, nargs='+')
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()

    output: pathlib.Path = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    # assert not output.exists(), f"File {output} already exists."

    df = pd.concat(
        pd.read_csv(csv, names=['id', 'completion_rate'], index_col='id')
        for csv in args.csvs
    )
    breakpoint()
    completion_rates = df['completion_rate'].groupby('id').max()

    todd.logger.info("Average completion rate: %f", completion_rates.mean())

    if todd.Store.DRY_RUN:
        return

    completion_rates.to_csv(output, header=False)


if __name__ == '__main__':
    main()
