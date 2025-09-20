.PHONY: build clean

PYTHON = python3
PIP = pip3

BUILD_DIR = build
BUILD_PII_DIR = $(BUILD_DIR)/pii
BUILD_PII_REMOTE = https://github.com/yamaceay/tab-anonymization.git

build:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	/Applications/Python\ 3.12/Install\ Certificates.command
	$(PYTHON) -m wn download oewn:2022

submodules: submodules.init

submodules.init:
	mkdir -p $(BUILD_DIR)
	git submodule init
	git submodule sync
	if [ -d $(BUILD_PII_DIR) ]; then git submodule update --remote --merge $(BUILD_PII_DIR); else git submodule add -f $(BUILD_PII_REMOTE) $(BUILD_PII_DIR); fi
	cd $(BUILD_PII_DIR) && \
		$(PIP) install --upgrade pip && \
		$(PIP) install -r requirements.txt && \
	cd ../..

submodules.clean:
	git submodule deinit -f build/pii
	git rm -f build/pii
	rm -rf .git/modules/build/pii

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete