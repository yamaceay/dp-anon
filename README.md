# DPMLM+PATCH: Differentially Private Selective Text Rewriting

## Setup
In this repository, you will find a `requirements.txt` file, which contains all necessary Python dependencies.

Firstly, clone the repository and install the required packages:

```bash
git clone https://github.com/yamaceay/dpmlm.git
cd dpmlm
pip install -r requirements.txt
```

Initialize the PII detection submodule:

```bash
make submodules
```

### 🔧 **Command Line Interface**

Use the command line tool for quick processing:

```bash
# Basic usage
python3 main.py -t dpmlm -e 1.0 "Your text here"

# With PII annotation
python3 main.py -t dpmlm --annotator=path/to/pii/model "Text with names"

# Using presets
python3 main.py --preset dpmlm_high_privacy "Sensitive text"

# List available options
python3 main.py --list-mechanisms
python3 main.py --list-presets
```