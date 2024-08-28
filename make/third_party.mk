include make/common.mk

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