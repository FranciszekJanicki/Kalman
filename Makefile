PROJECT_DIR := $(shell pwd)
THIRD_PARTY_DIR := ${PROJECT_DIR}/third_party
BUILD_DIR := ${PROJECT_DIR}/build
LIB_DIR := ${PROJECT_DIR}/lib
APP_DIR := ${PROJECT_DIR }/app

.PHONY: build
build: 
	cd ${BUILD_DIR} && make

.PHONY: run
run:
	cd ${BUILD_DIR} && ./App

.PHONY: clean
clean:
	cd ${PROJECT_DIR} && rm -rf ${BUILD_DIR}

.PHONY: cmake
cmake:
	cd ${PROJECT_DIR} && make clean && mkdir ${BUILD_DIR} && cmake -S . -B ${BUILD_DIR}


.PHONY: setup-eigen
setup-eigen:
	@if [ ! -d "${THIRD_PARTY_DIR}" ]; then \
	mkdir ${THIRD_PARTY_DIR}; \
	fi
	@if [ -d "${THIRD_PARTY_DIR}/eigen" ]; then \
	$(MAKE) eigen-remove; \
	fi
	cd ${THIRD_PARTY_DIR}
	git submodule add -f -b 3.4 https://gitlab.com/libeigen/eigen  ${THIRD_PARTY_DIR}/eigen

.PHONY: clean-eigen
clean-eigen:
	git submodule deinit -f $(THIRD_PARTY_DIR)/eigen
	rm -rf .git/modules/$(THIRD_PARTY_DIR)/eigen
	rm -rf $(THIRD_PARTY_DIR)/eigen
	git rm --cached ${THIRD_PARTY_DIR}/eigen

.PHONY: setup-external
setup-external:
	cd $(PROJECT_DIR) && touch .gitmodules && $(MAKE) setup-eigen


.PHONY: clean-external
clean-external: clean-eigen
	git submodule deinit --all
	rm -rf .gitmodules