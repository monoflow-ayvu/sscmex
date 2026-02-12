# Makefile wrapper for CMake-based NIF build

ifndef MIX_APP_PATH
	MIX_APP_PATH=$(shell pwd)
endif

PRIV_DIR = $(MIX_APP_PATH)/priv
SSCMEX_SO = $(PRIV_DIR)/sscmex_nif.so
CMAKE_BUILD_DIR = $(MIX_APP_PATH)/cmake_build

ifdef CMAKE_TOOLCHAIN_FILE
	CMAKE_CONFIGURE_FLAGS=-D CMAKE_TOOLCHAIN_FILE="$(CMAKE_TOOLCHAIN_FILE)"
endif

# Default build type
CMAKE_BUILD_TYPE ?= Release

# Default jobs
MAKE_JOBS ?= $(shell nproc 2>/dev/null || echo 1)

build: $(SSCMEX_SO)

$(SSCMEX_SO): $(PRIV_DIR)
	@ mkdir -p "$(CMAKE_BUILD_DIR)" && \
	cd "$(CMAKE_BUILD_DIR)" && \
	cmake --no-warn-unused-cli \
		-D CMAKE_BUILD_TYPE="$(CMAKE_BUILD_TYPE)" \
		-D C_SRC="$(shell pwd)/c_src" \
		-D MIX_APP_PATH="$(MIX_APP_PATH)" \
		-D PRIV_DIR="$(PRIV_DIR)" \
		$(CMAKE_CONFIGURE_FLAGS) "$(shell pwd)" && \
	cmake --build . --config "$(CMAKE_BUILD_TYPE)" -j$(MAKE_JOBS) && \
	cp "sscmex_nif.so" "$(SSCMEX_SO)"

clean:
	@rm -rf "$(CMAKE_BUILD_DIR)"
	@rm -f "$(SSCMEX_SO)"

$(PRIV_DIR):
	@mkdir -p "$(PRIV_DIR)"
