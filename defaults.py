from utils.LLM import ChatGPT
from functools import partial

# TODO: 将标注后的 clue-swc 测试集复制到
#       data/klg/clue-wsc/test.json
#       然后修改 clue - wsc 的值为 test.json
# TODO: 添加行测数据集
DefaultSubset = {
    "WMT19 - zh2en": "test.json",
    "WMT19 - en2zh": "test.json",
    "CSL - ctg": "test.tsv",
    "CSL - dcp": "test.tsv",
    "CSL - ts": "test.tsv",
    "Title2Event": "test.json",
    "AFQMC": "dev.json",
    "C3 - d": "test.json", 
    "C3 - m": "test.json",
    "NCR - xdw": "test.json",
    "CMRC": "dev.json",
    "VCSum - short": "test.txt",
    "math23k": "test.jsonl",
    "math401": "test.json",
    "LogiQA": "test.txt",
    "CHiD": "test.txt",
    "DRCD": "test.json",
    "NCR - gs": "test.json",
    "NCR - wyw": "test.json",
    "MSRA - NER": "test.txt",
    "clue - wsc": "dev.json"  # TODO: 修改为 test.json
}

DefaultLLMFunction = {
    "gpt-3.5-turbo": partial(ChatGPT.chat, model='gpt-3.5-turbo'),
    "gpt-4": partial(ChatGPT.chat, model='gpt-4'),
    "gpt-3.5-turbo-instruct": partial(ChatGPT.complete, max_tokens=3072)
}
