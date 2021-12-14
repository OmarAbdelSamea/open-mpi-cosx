#include <stdio.h>
#include <time.h>
#include <mpi.h>
#include <omp.h>

long double cos_calc_sign(size_t count)
{
    long double sign;
    if (count % 2)
        sign = -1;
    else
        sign = 1;
    return sign;
}

long double cos_calc_numerator(size_t count, long double x)
{
    long double numerator = 1;
    for (size_t i = 0; i < count; i++)
    {
        numerator *= x * x;
    }
    return numerator;
}

long double cos_calc_denominator(size_t count)
{
    long double f = 1;
    for (size_t i = 1; i <= 2 * count; i++)
        f = f * i;
    return f;
}

long double cos_calc_sequential(size_t k, long double x)
{
    long double sign = -1;
    long double numerator = 1;
    long double denominator = 1;
    long double result = 0;
    for (size_t i = 0; i <= k; i++)
    {
        sign = cos_calc_sign(i);
        numerator = cos_calc_numerator(i, x);
        denominator = cos_calc_denominator(i);
        result += (sign * numerator) / denominator;
    }
    return result;
}

long double cos_calc_parallel(size_t k, long double x, size_t rank, size_t number_processes)
{
    long double sign = -1;
    long double numerator = 1;
    long double denominator = 1;
    long double result = 0;
    for (size_t i = rank; i <= k; i = i + number_processes)
    {
        sign = cos_calc_sign(i);
        numerator = cos_calc_numerator(i, x);
        denominator = cos_calc_denominator(i);
        result += (sign * numerator) / denominator;
    }
    return result;
}

int main(int argc, char **argv)
{

    size_t k;
    int number_processes, rank;
    double time1, time2, duration, global;
    long double local_cos = 0;
    setbuf(stdout, NULL);

    long double x;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &number_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
    {
        printf("Please enter upper limit of k: ");
        scanf("%lu", &k);
        printf("Please enter value of x: ");
        scanf("%Lf", &x);
    }

    MPI_Bcast(&k, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&x, 1, MPI_LONG_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    time1 = MPI_Wtime();

    local_cos = cos_calc_parallel(k, x, rank, number_processes);

    long double global_cos;
    MPI_Reduce(&local_cos, &global_cos, 1, MPI_LONG_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        printf("result = %.20Lf \n", global_cos);
    }
    time2 = MPI_Wtime();
    duration = time2 - time1;

    MPI_Reduce(&duration, &global, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0)
    {
        printf("parallel runtime is %f\n", global);
        double t1 = clock();
        long double result = cos_calc_sequential(k, x);
        double t2 = clock();

        printf("sequential runtime is %f\n", (t2 - t1) / CLOCKS_PER_SEC);
    }

    MPI_Finalize();
}