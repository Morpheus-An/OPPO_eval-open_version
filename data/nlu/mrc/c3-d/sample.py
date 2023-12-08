import random
import json
random.seed(1234)

def load_json(p):
    with open(p, 'r') as f:
        return json.load(f)


def dump_json(obj, p):
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False)

ds = load_json('./train.json')
samples = random.sample(ds, k=50)
dump_json(samples, './icl_samples.json')
