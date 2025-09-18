.PHONY: build clean

PYTHON = python3
PIP = pip3

build:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	/Applications/Python\ 3.12/Install\ Certificates.command
	$(PYTHON) -m wn download oewn:2022

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete