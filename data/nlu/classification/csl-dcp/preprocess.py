from nltk.util import pr
import pandas as pd
import json


def load_json(p):
    with open(p, 'r') as f:
        return json.load(f)


def dump_json(obj, p):
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False)


d = pd.read_csv('./test.tsv', sep='\t', header=None)

l = list(d[2].unique())

print('，'.join(l))
