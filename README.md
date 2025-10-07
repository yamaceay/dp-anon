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

```bash
usage: main.py [-h] [--type {dpmlm,dpprompt,dpparaphrase,dpbart}] [--dpmlm-plus] [--dpmlm-annotator DPMLM_ANNOTATOR] [--dpmlm-risk {uniform,shap,greedy}]
               [--dpmlm-risk-model DPMLM_RISK_MODEL] [--dpmlm-pii-threshold DPMLM_PII_THRESHOLD] [--epsilon EPSILON] [--data_out DATA_OUT] [--config CONFIG]
               [--preset PRESET] [--list-mechanisms] [--list-presets] [--verbose]
               [text]

positional arguments:
  text                  Input text to process

options:
  -h, --help            show this help message and exit
  --type {dpmlm,dpprompt,dpparaphrase,dpbart}, -t {dpmlm,dpprompt,dpparaphrase,dpbart}
                        Type of DP mechanism to use
  --dpmlm-plus          Use 'plus' method with addition/deletion (DPMLM only)
  --dpmlm-annotator DPMLM_ANNOTATOR
                        Path to PII annotator model (mutually exclusive with non-uniform risk scoring)
  --dpmlm-risk {uniform,shap,greedy}, --dpmlm-risk-type {uniform,shap,greedy}
                        Explainability mode for DPMLM risk allocation
  --dpmlm-risk-model DPMLM_RISK_MODEL
                        Transformers model name or path for risk-aware scoring (required for SHAP/greedy)
  --dpmlm-pii-threshold DPMLM_PII_THRESHOLD
                        Score threshold for keeping PII predictions when process_pii_only is enabled
  --epsilon EPSILON, -e EPSILON
                        Privacy parameter epsilon
  --data_out DATA_OUT   Output directory for anonymized data
  --config CONFIG       Path to JSON configuration file
  --preset PRESET       Use preset configuration
  --list-mechanisms     List available mechanisms and exit
  --list-presets        List available presets and exit
  --verbose, -v         Enable verbose logging
```

To list available mechanisms, use the following command:

```bash
python3 main.py --list-mechanisms
```

To run the DPMLM mechanism with specific settings for example, use the following command structure:

```bash
cat data/TAB/splitted/test.json | jq '.[0].text' | python3 main.py \
    -e 100 \
    -t dpmlm \
    --dpmlm-risk-type greedy \
    --dpmlm-risk-model models/tri_pipelines/if_any \
    --dpmlm-annotator models/pii_detectors/if_any \
    --dpmlm-pii-threshold 0.99
```