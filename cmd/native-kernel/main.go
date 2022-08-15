package main

import (
	"log"
	"unsafe"

	cl "github.com/opencl-go/cl22"
)

func main() {
	log.Printf("OpenCL application starting up...\n")
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
			deviceName, _ := cl.DeviceInfoString(deviceID, cl.DeviceNameInfo)
			log.Printf("trying device '%s'\n", deviceName)
			context, err := cl.CreateContext([]cl.DeviceID{deviceID}, nil, cl.OnPlatform(platformID))
			if err != nil {
				log.Printf("could not create context: %v\n", err)
				continue
			}

			var execFlags cl.DeviceExecCapabilitiesFlags
			_, err = cl.DeviceInfo(deviceID, cl.DeviceExecutionCapabilitiesInfo, unsafe.Sizeof(execFlags), unsafe.Pointer(&execFlags))
			if err != nil {
				log.Printf("could not determine execution capabilities: %v\n", err)
				continue
			}
			if (execFlags & cl.ExecNativeKernel) == 0 {
				log.Printf("device cannot execute native kernels\n")
				continue
			}
			log.Printf("device has native kernel support, continuing\n")
			workWithContext(deviceID, context)
			_ = cl.ReleaseContext(context)
		}
	}
}

func workWithContext(device cl.DeviceID, context cl.Context) {
	commandQueue, err := cl.CreateCommandQueueWithProperties(context, device)
	if err != nil {
		log.Printf("could not create command-queue: %v\n", err)
		return
	}
	defer func() { _ = cl.ReleaseCommandQueue(commandQueue) }()

	log.Printf("running simple kernel...\n")
	runSimpleNativeKernel(commandQueue)
	log.Printf("running kernel with memory\n")
	runMemoryNativeKernel(context, commandQueue)
}

func runSimpleNativeKernel(commandQueue cl.CommandQueue) {
	var event cl.Event
	err := cl.EnqueueNativeKernel(commandQueue, func([]unsafe.Pointer) {
		log.Printf("CB: simple kernel\n")
	}, nil, nil, &event)
	if err != nil {
		log.Printf("could not start native kernel: %v\n", err)
		return
	}
	defer func() { _ = cl.ReleaseEvent(event) }()
	err = cl.WaitForEvents([]cl.Event{event})
	if err != nil {
		log.Printf("could not wait for events: %v\n", err)
		return
	}
	log.Printf("finished simple kernel\n")
}

func runMemoryNativeKernel(context cl.Context, commandQueue cl.CommandQueue) {
	inputDataA := []byte{10, 11, 12, 13, 14}
	inputA, err := cl.CreateBuffer(context, cl.MemReadOnlyFlag|cl.MemCopyHostPtrFlag|cl.MemHostNoAccessFlag, len(inputDataA), unsafe.Pointer(&inputDataA[0]))
	if err != nil {
		log.Printf("could not create input buffer A: %v\n", err)
	}
	defer func() { _ = cl.ReleaseMemObject(inputA) }()

	inputDataB := []byte{20}
	inputB, err := cl.CreateBuffer(context, cl.MemReadOnlyFlag|cl.MemCopyHostPtrFlag|cl.MemHostNoAccessFlag, len(inputDataB), unsafe.Pointer(&inputDataB[0]))
	if err != nil {
		log.Printf("could not create input buffer B: %v\n", err)
	}
	defer func() { _ = cl.ReleaseMemObject(inputB) }()

	output, err := cl.CreateBuffer(context, cl.MemWriteOnlyFlag|cl.MemHostReadOnlyFlag, len(inputDataA), nil)
	if err != nil {
		log.Printf("could not create output buffer: %v\n", err)
	}
	defer func() { _ = cl.ReleaseMemObject(output) }()

	var kernelEvent cl.Event
	err = cl.EnqueueNativeKernel(commandQueue, memoryKernel, []cl.MemObject{inputA, inputB, output}, nil, &kernelEvent)
	if err != nil {
		log.Printf("could not start native kernel: %v\n", err)
		return
	}
	defer func() { _ = cl.ReleaseEvent(kernelEvent) }()

	var resultEvent cl.Event
	result := make([]byte, len(inputDataA))
	err = cl.EnqueueReadBuffer(commandQueue, output, false, 0, uintptr(len(inputDataA)), unsafe.Pointer(&result[0]), []cl.Event{kernelEvent}, &resultEvent)
	if err != nil {
		log.Printf("could not read buffer: %v\n", err)
		return
	}

	err = cl.WaitForEvents([]cl.Event{resultEvent})
	if err != nil {
		log.Printf("could not wait for events: %v\n", err)
		return
	}
	log.Printf("finished memory kernel, data: %+v\n", result)
}

func memoryKernel(args []unsafe.Pointer) {
	log.Printf("CB: memory kernel\n")
	inputDataA := unsafe.Slice((*byte)(args[0]), 5)
	inputDataB := unsafe.Slice((*byte)(args[1]), 1)
	output := unsafe.Slice((*byte)(args[2]), 5)

	log.Printf("CB: inputDataA: %+v\n", inputDataA)
	log.Printf("CB: inputDataB: %+v\n", inputDataB)
	log.Printf("CB: oputput: %+v\n", output)

	for i := 0; i < len(inputDataA); i++ {
		output[i] = inputDataA[i] + inputDataB[0]
	}
}
