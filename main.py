import LLMDP
import DPMLM

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("text", type=str)
    parser.add_argument("--type", "-t", type=str, default="rewrite", choices=["rewrite", "rewrite-plus", "prompt", "paraphrase", "bart"])
    parser.add_argument("--epsilon", "-e", type=float, default=100)
    args = parser.parse_args()

    result = None
    if args.type in ["rewrite", "rewrite-plus"]:
        M = DPMLM.DPMLM()
        if args.type == "rewrite":
            result, *_ = M.dpmlm_rewrite(args.text, epsilon=args.epsilon)
        elif args.type == "rewrite-plus":
            result, *_ = M.dpmlm_rewrite_plus(args.text, epsilon=args.epsilon)

    elif args.type in ["prompt", "paraphrase", "bart"]:
        M = None
        if args.type == "prompt":
            M = LLMDP.DPPrompt()
        elif args.type == "paraphrase":
            M = LLMDP.DPParaphrase()
        elif args.type == "bart":
            M = LLMDP.DPBart()
        result = M.privatize(args.text, epsilon=args.epsilon)

    print(result)