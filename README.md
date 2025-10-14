# DPMLM+PATCH: Differentially Private Selective Text Rewriting

## Setup
In this repository, you will find a `requirements.txt` file, which contains all necessary Python dependencies.

Firstly, clone the repository and install the required packages:

```bash
git clone https://github.com/yamaceay/dpmlm.git
cd dpmlm
pip install -r requirements.txt
```

> For downloading Trustpilot dataset, please refer to the [Trustpilot dataset repository](https://www.kaggle.com/datasets/jerassy/trustpilot-reviews-123k/data) and save it as `data/trustpilot/trustpilot_reviews_2005.csv`.

Initialize the PII detection submodule:

```bash
make submodules
```

### 🔧 **Command Line Interface**

```bash
usage: main.py [-h] [--type {dpmlm,dpprompt,dpparaphrase,dpbart}] [--epsilon EPSILON]
               [--device DEVICE] [--seed SEED] [--config CONFIG] [--preset PRESET]
               [--data-out DATA_OUT] [--list-mechanisms] [--list-presets]
               [--verbose]
               [text]

positional arguments:
  text                  Optional input text; falls back to stdin or config runtime.input_text

options:
  -h, --help            show this help message and exit
  --type {dpmlm,dpprompt,dpparaphrase,dpbart}, -t {dpmlm,dpprompt,dpparaphrase,dpbart}
                        Type of DP mechanism to use
  --epsilon EPSILON, -e EPSILON
                        Privacy parameter epsilon used at runtime
  --device DEVICE       Device preference (auto, cpu, cuda, mps)
  --seed SEED           Random seed for mechanism initialisation
  --config CONFIG       Path to JSON configuration file containing model-specific parameters
  --preset PRESET       Use preset configuration and optionally override sections
  --data-out DATA_OUT   Output directory used when loading annotators
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
    --type dpmlm \
    --epsilon 8 \
    --device cuda \
    --config configs/config.json
```

### 🧪 Unified Benchmark Runner

For side-by-side evaluations across rule-based annotations, DP rewriting models, and PETRE variants, use the unified benchmark script:

```bash
python3 benchmark_unified.py --dataset tab --max-records 5 --methods manual spacy dpmlm_greedy_p095 dpprompt
```

Key defaults:
- `epsilon` defaults to `25.0`
- `tab` uses `data/TAB/splitted/train.json` when no dataset path is provided
- results are written under `outputs/unified_benchmark/<dataset>/<split>/benchmark_eps_25_0.json`
- `--methods list` prints all supported anonymisation options

Each method writes either token-level `annotations` or full `anonymized_text` pairs (original versus anonymised) so you can mix and match during evaluation. The helper `annotation_utils.apply_annotations(text, annotations)` converts any stored span format back into anonymised text at runtime for custom experiments.
