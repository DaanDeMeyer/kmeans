import csv
from argparse import ArgumentParser
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PARSER = ArgumentParser()
PARSER.add_argument("points_csv", type=str)
PARSER.add_argument("point_clusters_csv", type=str)


def main():
    args = PARSER.parse_args()
    points = []

    with open(args.points_csv) as points_csv:
        points_reader = csv.reader(points_csv)
        for point in points_reader:
            point = (float(point[0]), float(point[1]))
            points.append(point)

    with open(args.point_clusters_csv) as point_clusters_csv:
        clusters_reader = csv.reader(point_clusters_csv)
        point_clusters = next(clusters_reader)
        point_clusters = [int(x) for x in point_clusters]

    df = pd.DataFrame(columns=["cluster", "x", "y"])
    for index, point in enumerate(points):
        df.loc[index] = [point_clusters[index], point[0], point[1]]

    df["cluster"] = df["cluster"].astype(int)

    sns.lmplot("x", "y", data=df, hue="cluster", fit_reg=False)

    plt.show()


if __name__ == '__main__':
    main()
