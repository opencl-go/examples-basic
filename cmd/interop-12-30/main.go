package main

import (
	"log"
	"unsafe"

	"github.com/opencl-go/cl12"
	"github.com/opencl-go/cl30"
)

func main() {
	log.Printf("OpenCL 1.2 application starting up...\n")
	platformIDs, err := cl12.PlatformIDs()
	if err != nil {
		log.Fatalf("could not query platform IDs: %v\n", err)
	}
	if len(platformIDs) == 0 {
		log.Printf("no platform to work with\n")
		return
	}
	for _, platformID := range platformIDs {
		platformName, _ := cl12.PlatformInfoString(platformID, cl12.PlatformNameInfo)
		log.Printf("trying platform '%s'\n", platformName)
		deviceIDs, err := cl12.DeviceIDs(platformID, cl12.DeviceTypeAll)
		if err != nil {
			log.Printf("could not retrieve devices: %v\n", err)
			continue
		}
		if len(deviceIDs) == 0 {
			log.Printf("no devices on this platform\n")
			continue
		}
		for _, deviceID := range deviceIDs {
			deviceName, _ := cl12.DeviceInfoString(deviceID, cl12.DeviceNameInfo)
			log.Printf("trying device '%s'\n", deviceName)
			context, err := cl12.CreateContext([]cl12.DeviceID{deviceID}, nil, cl12.OnPlatform(platformID))
			if err != nil {
				log.Printf("could not create context: %v\n", err)
				continue
			}

			workWithContext(cl30.Context(context))

			_ = cl12.ReleaseContext(context)
		}
	}
}

func workWithContext(context cl30.Context) {
	log.Printf("OpenCL 3.0 library working with context...\n")
	err := cl30.RetainContext(context)
	if err != nil {
		log.Printf("failed to retain context: %v\n", err)
		return
	}
	defer func() { _ = cl30.ReleaseContext(context) }()

	var refCount uint32
	_, err = cl30.ContextInfo(context, cl30.ContextReferenceCountInfo, unsafe.Sizeof(refCount), unsafe.Pointer(&refCount))
	if err != nil {
		log.Printf("failed to retrieve reference count: %v\n", err)
	}
	log.Printf("reference count of context: %v\n", refCount)
}
