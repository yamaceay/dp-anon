import LLMDP
import DPMLM
from pii import (
    DataLabels,
    TorchTokenClassifier,
    PIIDeidentifier,
)

def recode_text(text):
    escape_sequences = {
        '\\n': '\n',
        '\\t': '\t',
        '\\r': '\r',
        '\\"': '"',
        "\\'": "'",
        '\\\\': '\\'
    }
    
    for escaped, unescaped in escape_sequences.items():
        text = text.replace(escaped, unescaped)
    return text

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("text", type=str)
    parser.add_argument("--type", "-t", type=str, default="rewrite", choices=["rewrite", "prompt", "paraphrase", "bart"])
    parser.add_argument("--plus", action="store_true")
    parser.add_argument("--annotator", type=str, default=None)
    parser.add_argument("--epsilon", "-e", type=float, default=100)
    parser.add_argument("--data_out", type=str, default='pii/outputs')
    args = parser.parse_args()

    args.text = recode_text(args.text)

    unique_labels = ['CODE', 'DEM', 'ORG', 'QUANTITY', 'LOC', 'DATETIME', 'MISC', 'PERSON']
    labels = ['O'] + [f'B-{label}' for label in unique_labels] + [f'I-{label}' for label in unique_labels]
    result = None
    perturbed, total = 0, 0
    if args.type == "rewrite":
        if args.annotator is not None:
            labels = DataLabels(labels)
            with TorchTokenClassifier(args.annotator, labels) as (model, tokenizer):
                annotator = PIIDeidentifier(args.data_out, model, tokenizer, labels)
                M = DPMLM.DPMLM(annotator=annotator)
                if not args.plus:
                    result, perturbed, total = M.dpmlm_rewrite_patch(args.text, epsilon=args.epsilon)
                else:
                    M.dpmlm_rewrite_patch_plus(args.text, epsilon=args.epsilon)
        else:
            M = DPMLM.DPMLM()
            if not args.plus:
                result, perturbed, total = M.dpmlm_rewrite(args.text, epsilon=args.epsilon)
            else:
                result, *_ = M.dpmlm_rewrite_plus(args.text, epsilon=args.epsilon)

    else:
        M = None
        if args.type == "prompt":
            M = LLMDP.DPPrompt()
        elif args.type == "paraphrase":
            M = LLMDP.DPParaphrase()
        elif args.type == "bart":
            M = LLMDP.DPBart()
        result = M.privatize(args.text, epsilon=args.epsilon)

    print('Original text:')
    print(repr(args.text))
    print()
    print('Result text:')
    print(repr(result))
    print()
    print(f'Perturbed: {perturbed}/{total} tokens')