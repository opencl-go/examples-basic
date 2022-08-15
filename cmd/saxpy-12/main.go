package main

import (
	_ "embed"
	"log"
	"unsafe"

	cl "github.com/opencl-go/cl12"
)

func main() {
	platformIDs, err := cl.PlatformIDs()
	if err != nil {
		log.Fatalf("could not query platform IDs: %v\n", err)
	}
	if len(platformIDs) == 0 {
		log.Printf("no platform to work with\n")
		return
	}
	for _, platformID := range platformIDs {
		platformName, _ := cl.PlatformInfoString(platformID, cl.PlatformNameInfo)
		log.Printf("trying platform '%s'\n", platformName)
		deviceIDs, err := cl.DeviceIDs(platformID, cl.DeviceTypeAll)
		if err != nil {
			log.Printf("could not retrieve devices: %v\n", err)
			continue
		}
		if len(deviceIDs) == 0 {
			log.Printf("no devices on this platform\n")
			continue
		}
		for _, deviceID := range deviceIDs {
			workWithDevice(platformID, deviceID)
		}
	}
}

func workWithDevice(platformID cl.PlatformID, deviceID cl.DeviceID) {
	deviceName, _ := cl.DeviceInfoString(deviceID, cl.DeviceNameInfo)
	log.Printf("trying device '%s'\n", deviceName)
	context, err := cl.CreateContext([]cl.DeviceID{deviceID}, nil, cl.OnPlatform(platformID))
	if err != nil {
		log.Printf("could not create context: %v\n", err)
		return
	}
	defer func() { _ = cl.ReleaseContext(context) }()

	commandQueue, err := cl.CreateCommandQueue(context, deviceID, 0)
	if err != nil {
		log.Printf("could not create command-queue: %v\n", err)
		return
	}
	defer func() { _ = cl.ReleaseCommandQueue(commandQueue) }()

	runSaxpy(deviceID, context, commandQueue)
}

//go:embed "saxpy.cl"
var saxpySource string

func runSaxpy(deviceID cl.DeviceID, context cl.Context, commandQueue cl.CommandQueue) {
	const dataSize = 1024
	alpha := float32(2.0)
	x := make([]float32, dataSize)
	y := make([]float32, dataSize)
	for i := 0; i < dataSize; i++ {
		x[i] = float32(i)
		y[i] = float32(dataSize - i)
	}

	// In the following, for brevity of the example, the code foregoes on error handling.

	aMemory, _ := cl.CreateBuffer(context, cl.MemReadOnlyFlag, dataSize*int(unsafe.Sizeof(float32(0))), nil)
	defer func() { _ = cl.ReleaseMemObject(aMemory) }()
	bMemory, _ := cl.CreateBuffer(context, cl.MemReadOnlyFlag, dataSize*int(unsafe.Sizeof(float32(0))), nil)
	defer func() { _ = cl.ReleaseMemObject(bMemory) }()
	cMemory, _ := cl.CreateBuffer(context, cl.MemWriteOnlyFlag, dataSize*int(unsafe.Sizeof(float32(0))), nil)
	defer func() { _ = cl.ReleaseMemObject(cMemory) }()

	_ = cl.EnqueueWriteBuffer(commandQueue, aMemory, true, 0, dataSize*unsafe.Sizeof(float32(0)), unsafe.Pointer(&x[0]), nil, nil)
	_ = cl.EnqueueWriteBuffer(commandQueue, bMemory, true, 0, dataSize*unsafe.Sizeof(float32(0)), unsafe.Pointer(&y[0]), nil, nil)

	program, _ := cl.CreateProgramWithSource(context, []string{saxpySource})
	defer func() { _ = cl.ReleaseProgram(program) }()
	_ = cl.BuildProgram(program, []cl.DeviceID{deviceID}, "", nil)
	kernel, _ := cl.CreateKernel(program, "saxpy_kernel")
	defer func() { _ = cl.ReleaseKernel(kernel) }()

	_ = cl.SetKernelArg(kernel, 0, unsafe.Sizeof(alpha), unsafe.Pointer(&alpha))
	_ = cl.SetKernelArg(kernel, 1, unsafe.Sizeof(aMemory), unsafe.Pointer(&aMemory))
	_ = cl.SetKernelArg(kernel, 2, unsafe.Sizeof(bMemory), unsafe.Pointer(&bMemory))
	_ = cl.SetKernelArg(kernel, 3, unsafe.Sizeof(cMemory), unsafe.Pointer(&cMemory))

	workDim := []cl.WorkDimension{
		{GlobalOffset: 0, GlobalSize: dataSize, LocalSize: 64},
	}
	_ = cl.EnqueueNDRangeKernel(commandQueue, kernel, workDim, nil, nil)

	z := make([]float32, dataSize)
	_ = cl.EnqueueReadBuffer(commandQueue, cMemory, true, 0, dataSize*unsafe.Sizeof(float32(0)), unsafe.Pointer(&z[0]), nil, nil)

	_ = cl.Flush(commandQueue)
	_ = cl.Finish(commandQueue)

	for i := 0; i < dataSize; i++ {
		log.Printf("%f * %f + %f = %f\n", alpha, x[i], y[i], z[i])
	}
}
