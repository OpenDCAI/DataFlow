# env
export HF_ENDPOINT=https://hf-mirror.com
export PATH=$PATH:/root/workspace/culfjk4p420c73amv510/herunming/DataFlow


# Step 1, Run static check for code data
echo  "\033[32m===== [Step 1] Static Check =====\033[0m"
python CodePipeline/code/static_check.py --yaml_path CodePipeline/yaml/static_check_test_case.yaml

python CodePipeline/code/code_comment.py --yaml_path CodePipeline/yaml/code_comment_test_case.yaml


