import json
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='argparse')
    parser.add_argument('--input_file_path', type=str,
                       default="/tf/orion.zou/dataset/aflow_v3_2/pair_wise/aflow_v3_2_paw_full_train_21785.json",
                       help="load_dataset_path")
    parser.add_argument('--output_file_path', type=str,
                       default="/tf/orion.zou/dataset/aflow_v3_2/pair_wise/aflow_v3_2_paw_full_train_21785_transformed.json",
                       help="upload_repo_id")
    return parser.parse_args()

def transform_dataset_from_file(input_file_path, output_file_path):
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
            # 提取 prompt
            prompt = entry["conversations"][0]["value"] if entry.get("conversations") else ""
            
            # 构建 chosen 和 rejected
            chosen = [
                {"content": entry["conversations"][0]["value"], "role": "user"},
                {"content": entry["chosen"]["value"], "role": "assistant"}
            ]
            rejected = [
                {"content": entry["conversations"][0]["value"], "role": "user"},
                {"content": entry["rejected"]["value"], "role": "assistant"}
            ]
            
            # 构建新的结构
            transformed.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected
            })
        
        # 保存转换后的数据到输出文件
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(transformed, f, indent=4, ensure_ascii=False)
        
        print(f"转换完成！数据已保存到 {output_file_path}")
    
    except FileNotFoundError as e:
        print(e)
    except json.JSONDecodeError:
        print(f"文件 {input_file_path} 不是有效的 JSON 文件。")
    except Exception as e:
        print(f"发生错误: {e}")

# 程序入口
if __name__ == "__main__":
    args = parse_args()
    # 调用转换函数
    transform_dataset_from_file(input_file_path=args.input_file_path, output_file_path=args.output_file_path)

