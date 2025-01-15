import os
import json
import time
import argparse
import requests
import numpy as np
from tqdm import tqdm
from itertools import chain


def parse_args():
    parser = argparse.ArgumentParser(description='vllm reward model config')
    parser.add_argument('--input_pairwise_file', type=str,
                       default="/tf/orion.zou/dataset/aflow_v3_2/pair_wise/aflow_v3_2_paw_full_test_2494.json",
                       help="input file")
    parser.add_argument('--url', type=str,
                       default="http://server1.zhaojunhua.org:89/v7/v1/reward",
                       help="url")
    parser.add_argument('--batch_size', type=int,
                       default=32,
                       help="host")

    return parser.parse_args()


def transform_rmdataset_from_file(input_file_path: str)-> dict:
    try:
        # 检查输入文件是否存在
        if not os.path.exists(input_file_path):
            raise FileNotFoundError(f"输入文件 {input_file_path} 不存在。")
        
        # 读取输入 JSON 文件
        with open(input_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # 转换数据
        transformed = []
        for entry in data:
            dataset_type = entry["conversations"][0]["dataset_type"] if entry.get("conversations") else ""
            # 提取 prompt
            query = entry["conversations"][0]["value"] if entry.get("conversations") else ""
            
            # 构建 chosen 和 rejected
            chosen_code = entry["chosen"]["value"]
            rejected_code = entry["rejected"]["value"]
            
            # 构建新的结构
            transformed.append({
                "dataset_type": dataset_type,
                "queries": [query,query],
                "codes": [chosen_code,rejected_code],
            })
        
        return transformed
    
    except FileNotFoundError as e:
        print(e)
    except json.JSONDecodeError:
        print(f"文件 {input_file_path} 不是有效的 JSON 文件。")
    except Exception as e:
        print(f"发生错误: {e}")

# 合并逻辑
def compress_rmlist(data, batch_size):
    result = []
    bs = int(batch_size/2)
    for i in range(0, len(data), bs):
        if len(data)>bs:
            tmp_data=data[i:i+bs]
        else:
            tmp_data=data[i:]
        result.append(
            {
                "dataset_type":tmp_data[0]["dataset_type"],
                "queries":list(chain.from_iterable([record["queries"] for record in tmp_data])),
                "codes":list(chain.from_iterable([record["codes"] for record in tmp_data])),
            }
        )  
    return result


def request_reward_api(url, data):
    headers = {
    "Content-Type": "application/json"
    }
    start_time =time.time()
    response = requests.post(url, headers=headers, json=data)
    print("Status Code:", response.status_code,f" Used time:{round(time.time()-start_time)}s")
    return [result["score"] for result in response.json()["data"]]

def split_list(original_list, chunk_size=2):
    nested_list = [original_list[i:i + chunk_size] for i in range(0, len(original_list), chunk_size)]
    return nested_list

def metrix(score_pair_list):
    chosens=[score_pair[0] for score_pair in score_pair_list]
    chosen = np.mean(chosens)
    print(f"rewards/chosen:{chosen}")
    rejecteds=[score_pair[1] for score_pair in score_pair_list]
    rejected = np.mean(rejecteds)
    print(f"rewards/rejected:{rejected}")
    bool_list = [score_pair[0] > score_pair[1] for score_pair in score_pair_list]
    acc = sum(bool_list) / len(score_pair_list)
    print(f"rewards/accuracies:{acc}")
    margins=[score_pair[0] - score_pair[1] for score_pair in score_pair_list]
    margin=np.mean(margins)
    print(f"rewards/margins:{margin}")
    return [chosen,rejected,acc,margin]

def main(args):
    rm_bench = transform_rmdataset_from_file(args.input_pairwise_file)
    # import ipdb;ipdb.set_trace()
    rm_bench =compress_rmlist(rm_bench, args.batch_size)

    score_list=[]
    for rm_data in tqdm(rm_bench, desc="Processing Data"):
        result = request_reward_api(args.url, rm_data)
        score_list.extend(result)

    score_pair_list = split_list(score_list,  chunk_size=2)
    print(f"acc:{metrix(score_pair_list)}")

args = parse_args()
main(args)