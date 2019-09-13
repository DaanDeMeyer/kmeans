#!/usr/bin/env python2
from os import path
import subprocess
import string
from argparse import ArgumentParser

try:
    from string import join
except ImportError:

    def join(x, y):
        return str.join(y, x)


INPUT_DIR = "input"
OUTPUT_DIR = "output"
RESULTS_DIR = path.join("benchmarks")

VERSIONS = ["mpi-group"]
NODES = [1]
CORES = [20]
FILES = ["100000"]
CLUSTERS = [60]
REPETITIONS = [75]

OMP_PLACES = "sockets"
I_MPI_PIN_DOMAIN = "socket"
OUTER_THREADS = 2
INNER_THREADS = 10

NUM_RUNS = 8
WALLTIME = "00:03:00"

VERSION_TO_COMMAND = {
    "seq": "seq",
    "omp-group": "omp-group",
    "omp-rep": "omp-rep",
    "mpi-group": "mpirun -hosts $HOSTS -n $NUM_PROCS -perhost $PERHOST mpi-group",
    "mpi-rep": "mpirun -hosts $HOSTS -n $NUM_PROCS -perhost $PERHOST mpi-rep",
}

OMP_ENV_TEMPLATE = string.Template("""export OMP_NESTED=TRUE
export OMP_PROC_BIND=close,master
export OMP_PLACES=${omp_places}
export OMP_NUM_THREADS=${outer_threads},${inner_threads}
""")

MPI_ENV_TEMPLATE = string.Template("""export HOSTS=`sort -u $$PBS_NODEFILE | paste -s -d,`
export I_MPI_PIN_DOMAIN=${i_mpi_pin_domain}
export PERHOST=${outer_threads}
export OMP_PROC_BIND=TRUE
export OMP_NUM_THREADS=${inner_threads}
export NUM_PROCS=${num_procs}
""")

DEFAULT_MODULES = """source switch_to_2015a
module load intel/2015a
"""

SINGLE_JOB_PER_NODE = "#PBS -l naccesspolicy=singlejob"

PBS_TEMPLATE = string.Template("""#!/bin/bash -l
#PBS -l ${partition}nodes=${nodes}${ppn}${cpu}
#PBS -l walltime=${walltime}
#PBS -A llp_h_pds
${single_job_per_node}

${modules}
${omp_env}${mpi_env}
cd $$PBS_O_WORKDIR

${command} --input ${input} --output ${output} --k ${clusters} --repetitions ${repetitions} >> ${result}
""")

PARSER = ArgumentParser()
PARSER.add_argument("-s", "--submit", action="store_true")


def main():
    args = PARSER.parse_args()

    for version in VERSIONS:
        for nodes in NODES:
            for cores in CORES:
                for filename in FILES:
                    input_path = path.join(INPUT_DIR, filename + ".csv")
                    output_path = path.join(OUTPUT_DIR, filename + ".csv")

                    for clusters in CLUSTERS:
                        for repetitions in REPETITIONS:
                            results_parts = [version, str(nodes), str(cores), filename, str(clusters), str(repetitions)]
                            results_name = join(results_parts, "-")
                            results_path = path.join(RESULTS_DIR, results_name)

                            with open("kmeans.pbs", "w") as pbs:
                                omp_env = ""
                                if version == "omp-group" or version == "omp-rep":
                                    omp_env = OMP_ENV_TEMPLATE.substitute(
                                        omp_places=OMP_PLACES, outer_threads=OUTER_THREADS, inner_threads=INNER_THREADS)

                                mpi_env = ""
                                if version == "mpi-group" or version == "mpi-rep":
                                    mpi_env = MPI_ENV_TEMPLATE.substitute(
                                        i_mpi_pin_domain=I_MPI_PIN_DOMAIN,
                                        outer_threads=OUTER_THREADS,
                                        inner_threads=INNER_THREADS,
                                        num_procs=OUTER_THREADS * nodes)

                                modules = DEFAULT_MODULES

                                partition = ""
                                ppn = ":ppn=" + str(cores)
                                cpu = ":ivybridge"
                                single_job_per_node = ""

                                script = PBS_TEMPLATE.substitute(
                                    partition=partition,
                                    nodes=nodes,
                                    ppn=ppn,
                                    cpu=cpu,
                                    walltime=WALLTIME,
                                    single_job_per_node=single_job_per_node,
                                    omp_env=omp_env,
                                    mpi_env=mpi_env,
                                    modules=modules,
                                    command=VERSION_TO_COMMAND[version],
                                    input=input_path,
                                    output=output_path,
                                    clusters=clusters,
                                    repetitions=repetitions,
                                    result=results_path)

                                pbs.write(script)

                            if args.submit:
                                for _ in range(0, NUM_RUNS):
                                    subprocess.call(["qsub", "kmeans.pbs"])


if __name__ == '__main__':
    main()
