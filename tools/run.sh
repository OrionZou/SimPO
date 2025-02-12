# export HF_ENDPOINT=https://hf-mirror.com
export all_proxy=http://127.0.0.1:1081

export HF_HOME=/tf/orion.zou/huggingface

# python tools/convert_codeUltreFeedback.py
# python tools/convert_aflow_dataset.py --input_file_path /tf/orion.zou/dataset/aflow_v6/pair_wise/aflow_v6_paw_train_4930.json --output_file_path /tf/orion.zou/dataset/aflow_v6/pair_wise/aflow_v6_paw_train_4930_tranformed.json
# python tools/convert_aflow_dataset.py --input_file_path /tf/orion.zou/dataset/aflow_v6/pair_wise/aflow_v6_paw_test_608.json --output_file_path /tf/orion.zou/dataset/aflow_v6/pair_wise/aflow_v6_paw_test_608_tranformed.json

python tools/convert_aflow_dataset.py --input_file_path /tf/orion.zou/dataset/aflow_v5/pair_wise/aflow_v5_paw_train_11889.json --output_file_path /tf/orion.zou/dataset/aflow_v5/pair_wise/aflow_v5_paw_train_11889_transformed.json
python tools/convert_aflow_dataset.py --input_file_path /tf/orion.zou/dataset/aflow_v5/pair_wise/aflow_v5_paw_test_1526.json   --output_file_path /tf/orion.zou/dataset/aflow_v5/pair_wise/aflow_v5_paw_test_1526_transformed.json  

python tools/convert_aflow_dataset.py --input_file_path /tf/orion.zou/dataset/aflow_v7/pair_wise/aflow_v7_paw_train_21246.json  --output_file_path /tf/orion.zou/dataset/aflow_v7/pair_wise/aflow_v7_paw_train_21246_transformed.json 
python tools/convert_aflow_dataset.py --input_file_path /tf/orion.zou/dataset/aflow_v7/pair_wise/aflow_v7_paw_test_1905.json    --output_file_path /tf/orion.zou/dataset/aflow_v7/pair_wise/aflow_v7_paw_test_1905_transformed.json 