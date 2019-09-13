cmake_minimum_required(VERSION 3.9)
project(kmeans LANGUAGES CXX)

option(OMP OFF)
option(MPI OFF)

include(cmake/kmeans.cmake)

kmeans_add_library(common OBJECT)
target_sources(
  common
  PRIVATE
    src/kmeans/args.cpp
    src/kmeans/CSVReader.cpp
    src/kmeans/CSVWriter.cpp
    src/kmeans/distance.cpp
    src/kmeans/divide.cpp
    src/kmeans/io.cpp
    src/kmeans/random.cpp
)

kmeans_add_executable(seq)
target_sources(
  seq
  PRIVATE
    src/kmeans/seq/data.cpp
    src/kmeans/seq/kmeans.cpp
    src/kmeans/seq/main.cpp
)

target_link_libraries(seq PRIVATE common)

if(OMP)
  find_package(OpenMP)

  kmeans_add_executable(omp-group)
  target_sources(
    omp-group
    PRIVATE
      src/kmeans/omp-group/data.cpp
      src/kmeans/omp-group/kmeans.cpp
      src/kmeans/omp-group/main.cpp
  )

  target_link_libraries(
    omp-group
    PRIVATE
      common
      OpenMP::OpenMP_CXX
  )

  kmeans_add_executable(omp-rep)
  target_sources(
    omp-rep
    PRIVATE
      src/kmeans/omp-rep/data.cpp
      src/kmeans/omp-rep/kmeans.cpp
      src/kmeans/omp-rep/main.cpp
  )

  target_link_libraries(
    omp-rep
    PRIVATE
      common
      OpenMP::OpenMP_CXX
  )
endif()

if(MPI)
  find_package(MPI)
  find_package(OpenMP)

  kmeans_add_executable(mpi-group)
  target_sources(
    mpi-group
    PRIVATE
      src/kmeans/mpi-group/data.cpp
      src/kmeans/mpi-group/kmeans.cpp
      src/kmeans/mpi-group/main.cpp
  )

  target_link_libraries(
    mpi-group
    PRIVATE
      common
      MPI::MPI_CXX
      OpenMP::OpenMP_CXX
  )

  kmeans_add_executable(mpi-rep)
  target_sources(
    mpi-rep
    PRIVATE
      src/kmeans/mpi-rep/data.cpp
      src/kmeans/mpi-rep/kmeans.cpp
      src/kmeans/mpi-rep/main.cpp
  )

  target_link_libraries(
    mpi-rep
    PRIVATE common
      MPI::MPI_CXX
      OpenMP::OpenMP_CXX
  )
endif()
