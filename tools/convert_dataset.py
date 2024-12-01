import json
import os

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
    # 示例输入文件路径和输出文件路径
    input_file = "/tf/orion.zou/dataset/aflow_v2/pair_wise/aflow_v2_paw_train_9475.json"  # 替换为实际输入文件路径
    output_file = "/tf/orion.zou/dataset/aflow_v2/pair_wise/aflow_v2_paw_train_9475_transformed.json"  # 替换为实际输出文件路径
    
    # 调用转换函数
    transform_dataset_from_file(input_file, output_file)

    input_file = "/tf/orion.zou/dataset/aflow_v2/pair_wise/aflow_v2_paw_test_671.json"  # 替换为实际输入文件路径
    output_file = "/tf/orion.zou/dataset/aflow_v2/pair_wise/aaflow_v2_paw_test_671_transformed.json"  # 替换为实际输出文件路径
    
    # 调用转换函数
    transform_dataset_from_file(input_file, output_file)