package main

import (
	"log"
	"unsafe"

	"github.com/opencl-go/cl22"
	"github.com/opencl-go/cl30"
)

func main() {
	log.Printf("OpenCL 3.0 application starting up...\n")
	platformIDs, err := cl30.PlatformIDs()
	if err != nil {
		log.Fatalf("could not query platform IDs: %v\n", err)
	}
	if len(platformIDs) == 0 {
		log.Printf("no platform to work with\n")
		return
	}
	for _, platformID := range platformIDs {
		platformName, _ := cl30.PlatformInfoString(platformID, cl30.PlatformNameInfo)
		log.Printf("trying platform '%s'\n", platformName)
		deviceIDs, err := cl30.DeviceIDs(platformID, cl30.DeviceTypeAll)
		if err != nil {
			log.Printf("could not retrieve devices: %v\n", err)
			continue
		}
		if len(deviceIDs) == 0 {
			log.Printf("no devices on this platform\n")
			continue
		}
		for _, deviceID := range deviceIDs {
			deviceName, _ := cl30.DeviceInfoString(deviceID, cl30.DeviceNameInfo)
			log.Printf("trying device '%s'\n", deviceName)
			context, err := cl30.CreateContext([]cl30.DeviceID{deviceID}, nil, cl30.OnPlatform(platformID))
			if err != nil {
				log.Printf("could not create context: %v\n", err)
				continue
			}

			workWithContext(cl22.Context(context))

			_ = cl30.ReleaseContext(context)
		}
	}
}

func workWithContext(context cl22.Context) {
	log.Printf("OpenCL 2.2 library working with context...\n")
	err := cl22.RetainContext(context)
	if err != nil {
		log.Printf("failed to retain context: %v\n", err)
		return
	}
	defer func() { _ = cl22.ReleaseContext(context) }()

	var refCount uint32
	_, err = cl22.ContextInfo(context, cl22.ContextReferenceCountInfo, unsafe.Sizeof(refCount), unsafe.Pointer(&refCount))
	if err != nil {
		log.Printf("failed to retrieve reference count: %v\n", err)
	}
	log.Printf("reference count of context: %v\n", refCount)
}
