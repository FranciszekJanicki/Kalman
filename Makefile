include make/third_party.mk

.PHONY: build
build: 
	cd ${BUILD_DIR} && make

.PHONY: run
run:
	cd ${BUILD_DIR} && ./Kalman.hex