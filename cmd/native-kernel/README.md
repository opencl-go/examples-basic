# Example: Native kernel

This example application shows how a native kernel can be run. Native kernels are executions on the host,
serialized as part of a command-queue.
Native kernels require a special execution capability of the device, so not every device will execute it.

If execution is possible, output may look something like this:

```
2022/08/15 07:52:50 OpenCL application starting up...
2022/08/15 07:52:50 trying platform 'NVIDIA CUDA'
2022/08/15 07:52:50 trying device 'NVIDIA GeForce GTX 1060'
2022/08/15 07:52:50 device cannot execute native kernels
2022/08/15 07:52:50 trying platform 'Intel(R) OpenCL'
2022/08/15 07:52:50 trying device 'Intel(R) HD Graphics 630'
2022/08/15 07:52:50 device cannot execute native kernels
2022/08/15 07:52:50 trying device 'Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz'
2022/08/15 07:52:50 device has native kernel support, continuing
2022/08/15 07:52:50 running simple kernel...
2022/08/15 07:52:50 CB: simple kernel
2022/08/15 07:52:50 finished simple kernel
2022/08/15 07:52:50 running kernel with memory
2022/08/15 07:52:50 CB: memory kernel
2022/08/15 07:52:50 CB: inputDataA: [10 11 12 13 14]
2022/08/15 07:52:50 CB: inputDataB: [20]
2022/08/15 07:52:50 CB: oputput: [0 0 0 0 0]
2022/08/15 07:52:50 finished memory kernel, data: [30 31 32 33 34]
```

The sample shows the basic execution of a native kernel, as well as the access to global-memory-mapped
buffers.
