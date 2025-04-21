import json
with open("/root/workspace/culfjk4p420c73amv510/herunming/DataFlow/100random_oss_inst_with_scores.jsonl", 'r') as f:
    data = [json.loads(_) for _ in f]
    
import random 
idx = random.randint(0, len(data))
for k, v in data[idx].items():
    print(f"{k}:")
    if isinstance(v, dict):
        for _, __ in v.items():
            print(f"{_}:")
            print(__)
    else:
        print(v)