# Example: SAXPY with OpenCL 1.2

This example application showcases the algebra operation "Single precision real Alpha X plus Y", short SAXPY.

The following shows equivalent Go code for the algorithm:
```go
const dataSize = 10
alpha := float32(2.0)
var x [dataSize]float32
var y [dataSize]float32
var z [dataSize]float32

for i := 0; i < dataSize; i++ {
    z[i] = alpha * x[i] + y[i]
}
```

If successful, the application may produce output similar to this:
```
2022/08/15 19:33:06 trying platform 'Intel(R) OpenCL HD Graphics'
2022/08/15 19:33:06 trying device 'Intel(R) Iris(R) Xe Graphics'
2022/08/15 19:33:07 2.000000 * 0.000000 + 1024.000000 = 1024.000000
2022/08/15 19:33:07 2.000000 * 1.000000 + 1023.000000 = 1025.000000
2022/08/15 19:33:07 2.000000 * 2.000000 + 1022.000000 = 1026.000000
2022/08/15 19:33:07 2.000000 * 3.000000 + 1021.000000 = 1027.000000
2022/08/15 19:33:07 2.000000 * 4.000000 + 1020.000000 = 1028.000000
2022/08/15 19:33:07 2.000000 * 5.000000 + 1019.000000 = 1029.000000
2022/08/15 19:33:07 2.000000 * 6.000000 + 1018.000000 = 1030.000000
...
```