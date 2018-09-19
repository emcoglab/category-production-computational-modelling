from os import path, getcwd
from os import remove as rm

from pandas import DataFrame, read_csv
from glob import glob


def main():
    root_dir = getcwd()
    for dir_path in glob(path.join(root_dir, "*")):
        if not path.isdir(dir_path):
            continue
        for csv_path in glob(path.join(dir_path, '*.csv')):
            if path.basename(csv_path) == "model_effectiveness.csv":
                rm(csv_path)
                continue
            print(f"Reading {csv_path}")
            # read comments
            comments = []
            with open(csv_path, mode="r", encoding="utf-8") as comment_file:
                for line in comment_file:
                    if line.startswith("#"):
                        comments.append(line)
            df: DataFrame = read_csv(csv_path, header=0, index_col=None, comment='#')
            df.sort_values(['Tick on which activated', 'Node ID'], inplace=True)
            print(f"Writing {csv_path}")
            with open(csv_path, mode="w", encoding="utf-8") as csv_file:
                for line in comments:
                    csv_file.write(line)
                df.to_csv(csv_file, header=True, index=False)


if __name__ == '__main__':
    main()
    print("Done")
