# kmeans

Multiple implementations of the kmeans algorithm. The following implementations
are provided:

- omp-group/mpi-group: OpenMP/MPI implementations that parallelize finding the
  nearest centroid for each point, the calculation of new centroids and
  calculating the cost of each solution of K-means.
- omp-rep/mpi-rep: OpenMP/MPI implementations with outer parallelism for
  dividing the repetitions of K-means and inner parallelism for calculating the
  nearest centroids and calculating the cost of each solution.
- seq: Sequential implementation of the K-means algorithm.

The MPI + OpenMP implementations should be configured to have a single process
per machine so OpenMP can be used to utilize all threads of that machine.

The rest of the README consists out of interesting sections from a set of
reports I wrote on these implementations.

## Vectorization

The above parallelizations should be sufficient to occupy all available cores on
most machines given any reasonably large workload (either in repetitions,
clusters, amount of points, dimension of points or any combination of these). To
further increase the performance of the algorithm we looked at another form of
parallelization, namely vectorization. Vectorization (or SIMD) allows a
processor core to execute a single instruction on multiple pieces of data. The
Ivybridge processors that are used by VSC make use of the AVX vector instruction
set. However, instead of directly using the AVX primitives made available by
compilers to vectorize our code, we make use of the OpenMP support for
vectorization (added in OpenMP 4.0 and conveniently supported by the 16.0
version of the Intel compiler, available in the 2016a software on VSC).

There are two primary targets for vectorization: Calculating the euclidian
distance and the summation part of calculating the new centroids. Both of these
operations perform operations on vectors (arrays of doubles) which are
independent and can be performed in parallel. In the case of the euclidian
distance, results have to be gathered in a single distance variable but OpenMP
conveniently supports this via reductions for SIMD.

However, vectorization is not as simple as just writing the OpenMP simd pragma.
Since vectorization works by loading more than one element of the vector (4
doubles in the case of the AVX instruction set) inside vectorization registers,
if the dimension of the vectors is not a multiple of the vector length the
remainder of the vector has to be executed in a normal scalar (non-vectorized)
loop. This has a negative effect on performance (if (dimension % vector length
== 3) then 3 scalar operations have to be executed at the end of each loop
instead of a single extra vector operation). To solve this problem we make sure
the dimension of the points is always a multiple of 4 (the amount of doubles
that fit in a AVX vector register (256 bits)) by padding each point with zeros
if necessary. These padded zeros have no effect on the euclidian distance or
centroid calculation ((0 - 0 = 0), (0 + 0 = 0)) and avoid having to execute
multiple scalar operations at the end of each loop if the dimension is not a
multiple of 4.

Another consideration when using SIMD is memory alignment. If the SIMD
equivalents of load/store assembly instructions are not executed on memory
aligned to the vector register length, they throw an exception (unless the
slower unaligned memory versions are used). To deal with this the Intel compiler
generates a scalar loop before the vectorized loop that performs scalar
operations on single elements until an aligned element (memory address is a
multiple of the vector register length (256 bits = 32 bytes)) is reached, at
which point execution switches to the vectorized loop. This scalar loop can be
avoided by making sure our array is aligned to 32 bytes in memory. Aligning is
not supported by malloc in C or std::vector in C++, but fortunately the Intel
compiler provides \_mm_malloc for allocating aligned memory. We can use
\_mm_malloc to allocate memory aligned at 32 bytes to make sure no scalar loop
needs to be executed before our vectorized loop.

However, instead of aligning our points and centroids arrays to 32 bytes, we
choose to align them to 64 bytes. This is because aligning to 64 bytes aligns
our arrays to not only the AVX vector registers, but also the L1 cache lines (on
x86 architecture). L1 is the cache closest to the CPU. Processors are most
efficient at loading/storing data when this data is aligned at 64 bytes (due to
a single load always loading 64 bytes of data into the L1 cache). We take full
advantage of the L1 cache (while still being aligned at 32 bytes as required for
vectorization) by aligning our arrays to 64 bytes.

More information (also source):
https://software.intel.com/en-us/articles/data-alignment-to-assist-vectorization
Talk about impact of memory on performance (2nd source):
https://channel9.msdn.com/Events/Build/2013/4-329

## OpenMP

The insights gained during the implementation of k-means with MPI led us to
reconsider the choices made in the OpenMP implementation. In the OpenMP
implementation we made the choice to parallelize the repetitions of the k-means
algorithm. Our reasoning was that each core could be fully utilized as long as
there were sufficient repetitions and almost no locking would be required except
when updating the final result after each repetition. We later discovered a few
problems with this reasoning:

- The input is more likely to scale up in the amount of points, the dimension of
  the points and the amount of clusters (in that order). Having more cores
  available than repetitions is not an impossible scenario. However, we
  circumvented this somewhat by using the remaining cores to help other
  repetitions complete earlier using OpenMP nested parallelism.
- Each thread working on a separate repetition needs arrays to store
  intermediate clusters, centroids, cluster sizes, etc. Especially the centroids
  array can be several MB's in size when the amount of clusters is big and the
  dimension of the input data is large. As an example, storing the centroids
  when clustering drivFaceD.csv in 100 clusters requires a 4MB array per thread
  to store the centroids. Calculating 20 repetitions in parallel results in 80
  MB of arrays for storing centroids. This leads to many L3 cache misses while
  running the program (each ivybridge node only has 50 MB of L3 cache) which has
  a negative impact on performance that increases with the amount of threads
  used.

To solve the second problem we looked at parallelizing parts of the computation
that required a constant amount of memory regardless of the amount of threads
used. Using Intel VTune we also profiled the code to find the hotspots which
would benefit most from parallelization. As expected, computing the nearest
centroid for each point takes by far the most time in the k-means algorithm
(more than 90% of the time is spent here if the amount of clusters is high). The
second biggest hotspot was adding the points of each cluster together when
calculating the new centroids.

Computing the nearest centroid for each point can easily be parallelized by
evenly dividing the points between the threads and having each thread
calculating the closest centroid for its assigned point. Only read-only data is
shared which means no locking is necessary. This loop can be considered
embarassingly parallel and parallelizing it does not require us to allocate
additional memory per thread which allows the parallelization to scale to any
number of threads without dramatically increasing the number of cache misses.

Parallelizing the addition of the points of each cluster is harder. Here, if we
assign each thread part of the points, threads will be writing to shared data
and data races will occur. To solve this we tried assigning each thread a set of
clusters to update instead of a set of points to eliminate writing to shared
data but in practice this resulted in worse performance for some datasets while
not providing substantial benefits for other datasets so we decided against
parallelizing the addition of points when calculating the new centroids. We did
parallelize the calculation of the total cost at the end of each repetition
since again there is no shared data that needs to be written to which makes it a
prime target for parallelization (it also loops over all points which will in
most cases ensure there is enough work for all threads).

These new parallelizations do not require extra memory to be allocated per
thread. As a result, the amount of cache misses is lower compared to the MPI
implementation when the centroid array is large (MPI copies centroid array to
each process) which results in OpenMP being faster than MPI on a single node if
the centroid array is large. However, we noticed that the new OpenMP
implementation was slower by a factor of 2 compared to the MPI implementation on
datasets where all of the MPI centroid arrays (as well as the other data) could
completely fit into the L3 cache.

To understand why MPI is faster than OpenMP in this case we have to remember
that the single node we are using actually has 2 processors with 10 threads
instead of a single processor with 20 threads. When multiple processors are used
in a system, processors and memory use the NUMA (Non Uniform Memory Access) to
communicate. Under NUMA each processor is assigned part of the available memory
which is the local memory for that processor. Accessing its own local memory is
faster for a processor compared to accessing the local memory of another
processor (Remote Memory Access). Under NUMA, cache coherency algorithms are
used to keep the caches of all the processors in sync. With the small amount of
research we did into NUMA and cache coherence, we assumed (this may not be
entirely correct) that fewer remote memory/cache accesses for a processor would
reduce the overhead of RMA and cache coherence, as well as reducing the amount
of duplicated data in the processor caches (Remote memory/cache access causes
the processor to duplicate the accessed data in its own cache, resulting in each
cache having a copy of the data). Making more effective use of each processor's
cache (by not duplicating) and performing fewer RMA's should result in increased
performance.

To implement this, we changed the OpenMP implementation to behave more like the
MPI implementation. However, instead of dividing the points between threads and
having separate centroid, point clusters and cluster sizes arrays per thread, we
divide the points between processors and only have a separate centroid, point
clusters and cluster sizes array per processor. This reduces the amount of
centroid (and point clusters, cluster sizes) arrays per processor to 1 instead
of 10 while vastly reducing the amount of remote memory/cache accesses required
since the second processor has its own set of points and the required arrays
stored in its own local memory instead of the local memory of the first
processor. Communication between the processors only needs to happen when
combining the results of the processors, which happens at the same points as in
the MPI implementation (broadcasting the centroids before calculating nearest
centroids, reducing whether the point clusters are equal after calculating the
nearest centroids, retrieving the partial sums and cluster sizes when
calculating the new centroids and retrieving the actual point clusters when the
current repetition is done). Our hypothesis seems to be correct, since the NUMA
aware OpenMP implementation has almost exactly the same performance as the MPI
implementation (while still vastly outperforming the MPI implementation when the
centroid arrays are large since it only has 2 centroid arrays instead of 20).

For this assignment, we replicated the behaviour of the MPI implementation in
the OpenMP implementation to solve the NUMA and cache coherence issues since we
weren't allowed to use MPI and OpenMP together. If this was allowed, we would
just run a single MPI process per processor and not make any changes to the
OpenMP code. This might even be faster since we would be able to use MPI
reductions instead of just adding the results of all processors together in a
for loop.

In order for the OpenMP code to perform well, a few OpenMP environment variables
have to be set:

- OMP_NESTED: TRUE
- OMP_PLACES: sockets
- OMP_PROC_BIND: spread,master
- OMP_NUM_THREADS: 2,10

If multiple arguments are provided to proc bind, these correspond to different
levels of nested parallelism. If we are not in a parallel region yet, the spread
policy will be used. If we are already in a parallel region, the master policy
will be used. The same applies to the num threads variable.

Setting the OpenMP places to sockets, the first argument to proc bind to spread
and the first argument to num threads to 2 results in a non-nested parallel
region to use a single thread of each of the two processors. Each of these
threads stores half the points and the required arrays in the local memory of
the processor.

The second parameters of proc bind and num threads (master and 10) are used to
allow all cores of the processor to work on the data stored in the local memory
of the processor. All parallel regions start with a single thread of each
processor, with sometimes a nested parallel being used to allow each processor
to use all its cores to do work.

## MPI and OpenMP

In the previous report we dismissed parallelizing the repetitions of K-means
with MPI since it required each process to have its own copy of the input points
which would have a negative performance impact due to the limited cache size.
However, we didn't consider that the communication overhead of parallelizing the
repetitions is almost negligible compared to the communication overhead of
parallelizing the inner loops of the K-means algorithm. The communication
overhead of our current solution increases sharply with the amount of nodes
used, where in theory the communication overhead when parallelizing the
repetitions should hardly increase at all since only minimal communication is
required at the end of the algorithm.

The disadvantage of parallelizing the repetitions with MPI (and in general) is
that we can't parallelize any further once we run out of repetitions. To
circumvent this problem we combine multiple solutions. We use MPI to divide the
repetitions of K-means between multiple processors and in each of these
processors we use OpenMP to parallelize the calculation of single repetitions.
While spreading the calculation of a single repetition over multiple nodes
suffers from communication overhead, this overhead should be negligible inside a
single processor since every thread shares the data and there is no overhead
from copying or NUMA (as discussed in the previous report).

In the current implementation, the division of repetitions over processors is
hard-coded. A more complete solution would allow the user to specify how many
nodes/processors/cores should be allocated to a single repetition through the
use of environment variables. This would allow any number of nodes to be used
even if many nodes are available and the amount of repetitions is minimal.
However, we leave this for future work since the implementation would be rather
complex and our tests are performed with minimal nodes and more than enough
repetitions to ensure each node has at least a single repetition to calculate.

To allocate a single MPI process for each processor we make use of the
I_MPI_PIN_DOMAIN environment variable as described in the VSC documentation for
hybrid MPI/OpenMP programs.

![](images/mpi_vs_omp_inner.png?raw=true "Execution times with inner level parallelism")
![](images/mpi_vs_omp_outer.png?raw=true "Execution times with outer level parallelism")

From the test results we can see that there are only small differences between
the OpenMP implementations and their respective MPI counterparts. This is
expected since we programmed the MPI implementations to be extensions of the
OpenMP implementations that can execute on multiple nodes.

Another insight is that the outer level parallelism configurationperforms
slightly faster on 100000.csv but much slower on drivFaceD.csv compared to the
inner level parallelism configuration. A possible explanation is that outer
level parallelism uses significantly more memory than inner level parallelism
since a centroid array is allocated and used repeatedly for each outer level
thread/process. This leads to significantly more cache misses when the centroid
array is big (drivFaceD.csv) which explains the slowdown with drivFaceD.csv. On
the other hand, when the extra centroid arrays do fit into the cache
(100000.csv), the execution time can decrease since calculating the new
centroids happens in parallel by the outer level threads/processes (each
thread/process has a centroid array which avoids data races and a reduction
happens to combine the results of all outer threads/processes).

![](images/mpi_nodes_omp.png?raw=true "Execution times of both MPI implementations with inner level parallelism")

If we look at how the MPI implementations with OpenMP scale, we see that both
implementations scale very well to more than one node, with the extra overhead
of communication the centroids each iteration not having a significant impact on
the performance of mpi_group. Looking only at communication overhead we might
think parallelizing the repetitions would be faster since communication only
happens at the end of the program execution. However, since mpi_rep loads and
uses the entire input on each process (once per processor) which leads to many
more cache misses compared to mpi_group which divides the input over all the
nodes. We did not test with a sufficient amount of nodes to provide a conclusive
answer as to which implementation scales better.

## Test Setup

The OpenMP and MPI implementations are tested in two configurations which we
call outer level parallelism and inner level parallelism:

- omp-rep/mpi-rep:

  - Inner level parallelism:

    A single thread/process is allocated per processor which receives part of
    the repetitions (and the entire input). Each of these threads/processes has
    access to 10 threads which they use to parallelize finding the nearest
    centroids and calculating the cost of each of its assigned repetitions.

  - Outer level parallelism:

    Same as above except a single thread/process is allocated per core and no
    extra inner threads are available to each outer thread.

- omp-group/mpi-group:

  - Inner level parallelism:

    A single thread/process is allocated per processor which receives part of
    the input (and all repetitions) and calculates the nearest centroids,
    centroid sums and cost for that part of the input (for each repetition).
    Each of these threads/processes has access to 10 threads which they use to
    parallelize finding the nearest centroids and calculating the cost of their
    part of the input. The total centroid sum is reduced over all outer
    threads/processes and distributed back to each outer thread/process.

  - Outer level parallelism:

    Same as above except a single thread/process is allocated per core and no
    extra inner threads are available to each outer thread.

Relevant environment variables:

- OpenMP:
  - OMP_NESTED
  - OMP_PROC_BIND
  - OMP_PLACES
  - OMP_NUM_THREADS
- MPI (Intel MPI specific):
  - I_MPI_PIN_DOMAIN
  - I_MPI_PERHOST
