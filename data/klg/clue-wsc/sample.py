import random
import json
random.seed(1234)


def load_json(p):
    with open(p, 'r') as f:
        return json.load(f)


def dump_json(obj, p):
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


with open('./train.json', 'r') as f:
    lines = f.readlines()

samples = random.sample(lines, k=50)
with open('./icl_samples.json', 'w') as f:
    f.writelines(samples)
