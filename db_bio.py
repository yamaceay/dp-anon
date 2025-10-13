from datasets import load_dataset

from pathlib import Path

# dataset = load_dataset("arrow", data_files={"train": "./data/db_bio/train/data-00000-of-00001.arrow", "test": "./data/db_bio/test/data-00000-of-00001.arrow", "validation": "./data/db_bio/validation/data-00000-of-00001.arrow"})

# with open("./data/db_bio/train.jsonl", "w") as f:
#     for item in dataset["train"]:
#         f.write(f"{item}\n")

# with open("./data/db_bio/test.jsonl", "w") as f:
#     for item in dataset["test"]:
#         f.write(f"{item}\n")

# with open("./data/db_bio/validation.jsonl", "w") as f:
#     for item in dataset["validation"]:
#         f.write(f"{item}\n")