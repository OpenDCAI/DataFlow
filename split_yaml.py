import yaml

with open('qrt.yaml', 'r') as f:
    basic_cfg = yaml.safe_load(f)
    
with open('text_pipeline/yaml/raw_sft_filter.yaml', 'r') as f:
    all_cfg = yaml.safe_load(f)
    
with open("text_pipeline/raw_sft_filter.sh", 'w') as f:
    file_base = basic_cfg['input_file'][:-6]
    last_step = None
    for k, v in all_cfg['processors'].items():
        merged_dict = basic_cfg | v
        if last_step is None:
            merged_dict['input_file'] = file_base + ".jsonl"
        else:
            merged_dict['input_file'] = file_base + "_" + last_step + ".jsonl"
        merged_dict['output_file'] = file_base + "_" + k + ".jsonl"
        last_step = k
        with open(f"text_pipeline/yaml/{k}.yaml", 'w') as g:
            yaml.dump(merged_dict, g)
        f.write(f"python pipeline_step.py --yaml_path text_pipeline/yaml/{k}.yaml --step_name {k} --step_type process")
        f.write('\n')
        f.write('\n')
