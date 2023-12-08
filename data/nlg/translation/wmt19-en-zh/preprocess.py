from bs4 import BeautifulSoup
import json

def load_json(p):
    with open(p, 'r') as f:
        return json.load(f)


def dump_json(obj, p):
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False)

def get_texts(p):
    with open(p, 'r') as f:
        t = f.read()
    soup = BeautifulSoup(t, 'html.parser')
    segs = soup.find_all('seg')
    return [s.get_text() for s in segs]

def build(ref_p, src_p):
    ref_texts = get_texts(ref_p)
    src_texts = get_texts(src_p)
    ds = []
    for r,s in zip(ref_texts, src_texts):
        ds.append({'ref':r, 'src':s})
    return ds

# dev
ref_p = './newstest2018-enzh-ref.zh.sgm'
src_p = './newstest2018-enzh-src.en.sgm'
ds = build(ref_p, src_p)
print(len(ds))
dump_json(ds, './en-zh-dev.json')


# test
ref_p = './newstest2019-enzh-ref.zh.sgm'
src_p = './newstest2019-enzh-src.en.sgm'
ds = build(ref_p, src_p)
print(len(ds))
dump_json(ds, './en-zh-test.json')

