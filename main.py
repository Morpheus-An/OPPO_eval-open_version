from commons.common_import import BOLD_GREEN, RESET_COLOR
from datasets import ShowAvailableDatasets, AvailableDatasets
import argparse
from defaults import *

# ShowAvailableDatasets()
import openai 
import numpy as np
# print(np.array())

AvailableDatasetNames = list(DefaultSubset.keys())
AvailableLLMs = list(DefaultLLMFunction.keys())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, choices=AvailableLLMs,
                        default='gpt-3.5-turbo', help="model name")
    parser.add_argument("-d", "--dataset", type=str, choices=AvailableDatasetNames,
                        default=None, help="dataset name")
    parser.add_argument("-s", "--subset", type=str,
                        default=None, help="the subset to evaluate on")
    parser.add_argument("-n", "--sample_num", type=int,
                        default=None, help="number of samples to use")
    parser.add_argument("-ns", "--n_shot", type=int,
                        default=0, help="number of ICL samples")
    parser.add_argument("-f", "--icl_sample_file", type=str,
                        default=None, help="the file which stores the ICL samples")
    args = parser.parse_args()
    return args


def main():
    """main function"""
    args = parse_args()
    llm_func = DefaultLLMFunction[args.model]
    if args.subset is None:
        args.subset = DefaultSubset[args.dataset]
    print(f"{BOLD_GREEN}\nDataset: {args.dataset}\n" +
          f"Model:   {args.model}\n" +
          f"Arguments:\n{RESET_COLOR}", args, "\n")
    AvailableDatasets[args.dataset].evaluate(
        llm_func=llm_func,
        llm_name=args.model,
        subset_name=args.subset,
        sample_num=args.sample_num,
        icl_sample_file=args.icl_sample_file,
        n_shot=args.n_shot,
    )


if __name__ == '__main__':
    # exit()
    main()
