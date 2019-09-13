from os.path import join
from os import walk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

COLUMNS = [
    "version", "nodes", "cores", "file", "clusters", "repetitions", "time"
]
COLUMN_TYPES = [str, int, int, str, int, int, float]
INPUT_DIR = join("benchmarks")
OUTPUT_DIR = join("report", "plots")
VERSIONS_ORDER = ["omp-group", "mpi-group", "omp-rep", "mpi-rep"]

sns.set_style("whitegrid")


def main():
    mpi_nodes_omp = read_data("mpi_nodes_omp")
    nodes_compare(mpi_nodes_omp)
    save("mpi_nodes_omp")

    mpi_vs_omp_inner = read_data("mpi_vs_omp_inner")
    version_compare(mpi_vs_omp_inner)
    save("mpi_vs_omp_inner")

    mpi_vs_omp_outer = read_data("mpi_vs_omp_outer")
    version_compare(mpi_vs_omp_outer)
    save("mpi_vs_omp_outer")


def nodes_compare(df):
    sns.pointplot(x="nodes", y="time", hue="version", data=df)
    plt.xlabel("Nodes")
    plt.ylabel("Average Execution Time (s)")


def version_compare(df):
    sns.factorplot(
        x="version",
        y="time",
        hue="file",
        kind="bar",
        data=df,
        order=VERSIONS_ORDER)
    plt.xlabel("Version")
    plt.ylabel("Average Execution Time (s)")


def clusters_compare(df):
    sns.pointplot(x="clusters", y="time", hue="file", data=df)
    plt.xlabel("Clusters")
    plt.ylabel("Average Execution Time (s)")


def repetitions_compare(df):
    sns.pointplot(x="repetitions", y="time", hue="file", data=df)
    plt.xlabel("Repetitions")
    plt.ylabel("Average Execution Time (s)")


def read_data(results_dir):
    df = pd.DataFrame(columns=COLUMNS)

    complete_dir = join(INPUT_DIR, results_dir)

    (_, _, filenames) = next(walk(complete_dir))

    for filename in filenames:
        params = filename.split("-")
        with open(join(complete_dir, filename)) as results_file:
            execution_times = results_file.readlines()
            execution_times = [float(x) for x in execution_times]

            for time in execution_times:
                row = params[:]
                row.append(time)
                df.loc[len(df)] = row

    for column, column_type in zip(COLUMNS, COLUMN_TYPES):
        df[column] = df[column].astype(column_type)

    return df


def save(name):
    plt.savefig(
        join(OUTPUT_DIR, name + ".png"),
        dpi=150,
        format="png",
        bbox_inches="tight")
    plt.clf()


if __name__ == '__main__':
    main()
