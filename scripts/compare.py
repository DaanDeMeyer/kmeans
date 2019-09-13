import csv
from argparse import ArgumentParser

PARSER = ArgumentParser()
PARSER.add_argument("result_one_path", type=str)
PARSER.add_argument("result_two_path", type=str)


def main():
    args = PARSER.parse_args()

    cluster_map = {}

    with open(args.result_one_path) as result_one_csv, open(args.result_two_path) as result_two_csv:
        result_one_reader = csv.reader(result_one_csv)
        result_two_reader = csv.reader(result_two_csv)

        for one_cluster, two_cluster in zip(next(result_one_reader), next(result_two_reader)):
            if (not one_cluster in cluster_map.keys()) and (not two_cluster in cluster_map.values()):
                cluster_map[one_cluster] = two_cluster
                continue

            if (one_cluster in cluster_map.keys()) and (not two_cluster in cluster_map.values()):
                return 1

            if (not one_cluster in cluster_map.keys()) and (two_cluster in cluster_map.values()):
                return 1

            if cluster_map[one_cluster] != two_cluster:
                return 1

    return 0


if __name__ == '__main__':
    main()
