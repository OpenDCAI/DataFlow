import json
import os.path


def read_json(source):
    json_list = []
    if not os.path.exists(source):
        return json_list
    with open(source, 'r', encoding='utf-8') as f:
        for line in f:
            json_list.append(json.loads(line))
    return json_list


def dump_json(source, datas):
    with open(source, 'w', encoding='utf-8') as f:
        for item in datas:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

def load_file(load_path):
    with open(load_path, 'r', encoding='utf-8') as f1:
        data = json.load(f1)
        print(data[0])
    return data