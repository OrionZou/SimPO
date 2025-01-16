import json
import os
import re
import argparse
import itertools
import pandas as pd
from huggingface_hub import HfApi
from huggingface_hub import login
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk

def parse_args():
    parser = argparse.ArgumentParser(description='argparse')
    parser.add_argument('--load_dataset_path', type=str,
                       default="RLHFlow/CodeUltraFeedback-standard",
                       help="load_dataset_path")
    parser.add_argument('--upload_repo_id', type=str,
                       default="RLHFlow/CodeUltraFeedback-standard",
                       help="upload_repo_id")
    parser.add_argument('--upload_huggingface_token', type=str,
                       default="RLHFlow/CodeUltraFeedback-standard",
                       help="load_dataset_path")
    parser.add_argument('--pattern', type=str,
                       default="",
                       help="pattern")
    return parser.parse_args()

def convert_rlhfflow_codeUltraFeedback(load_dataset_path="RLHFlow/CodeUltraFeedback-standard",pattern=""):
    ds = load_dataset(load_dataset_path, split="train")
    data = []
    for example in ds:
        rejected_message = example['rejected']
        chosen_message = example['chosen']
        rejected_rating = example['rejected_score']
        chosen_rating = example['chosen_score']
        prompt = example['chosen'][0]["content"]
        if re.search(pattern, prompt, re.IGNORECASE):
            data.append({"prompt": prompt,"rejected": rejected_message, "chosen": chosen_message, "rejected_score": rejected_rating, "chosen_score": chosen_rating})
    print(len(data))
    return data


def upload_huggingface_hub(hf_dataset, repo_id="gyzou/codeUltraFeedback-pairwise", token="<huggingface_token>"):
    login(token=token)
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    hf_dataset.push_to_hub(repo_id)

    print(f"数据集已上传到 https://huggingface.co/datasets/{repo_id}")

# 程序入口
if __name__ == "__main__":
    args= parse_args()
    data_list=convert_rlhfflow_codeUltraFeedback(args.load_dataset_path, args.pattern)
    datasetdata = Dataset.from_list(data_list)

    train_test_split = datasetdata.train_test_split(test_size=0.05)

    # 将其包装成 DatasetDict
    hf_dataset = DatasetDict({
        "train": train_test_split["train"],
        "test": train_test_split["test"]
    })
    upload_huggingface_hub(hf_dataset,repo_id=args.upload_repo_id, token=args.upload_huggingface_token)

