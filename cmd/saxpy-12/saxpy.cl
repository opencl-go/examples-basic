__kernel
void saxpy_kernel(float alpha, __global float *x, __global float *y, __global float *z)
{
    int index = get_global_id(0);
    z[index] = alpha * x[index] + y[index];
}
