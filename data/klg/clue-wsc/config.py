from commons.common_import import *
global ParentDataset
EvalFuncDict.clear()


TaskDescription = """\
请你判断所给句子中，<< >> 包围的代词是不是指代 [[ ]] 包围的名词。\
如果你认为是，输出“是”，不然输出“否”。\
你只应该输出“是”或者“否”，不要输出其他字符。
"""
# TaskDescription = """\
# 请你判断所给句子中，<< >> 包围的代词是不是指代 [[ ]] 包围的名词。\
# 请你先说明理由，在那之后，如果你认为是，输出“答案：是”，不然输出“答案：否”。\
# """
QuestionPrompt = """\
句子：
{sentence}
答案：
{answer}
"""


class Dataset(ParentDataset):
    Description = {
        "name": "clue - wsc"
    }

    def analyse_file(self, fp: TextIO, sample_num: Optional[int] = None) -> List[Tuple[Any, Any]]:
        if sample_num is not None:
            assert type(sample_num) == int and sample_num > 0
        ds = [json.loads(l) for l in fp.readlines()]
        _dataset = []
        for d in ds:
            ans = (d["label"] == 'true') if "label" in d else ''
            _dataset.append((d, ans))
        # for d in _dataset[0:10]:
            # print(d)
        return _dataset

    def fill_prompt(self, sample: Any, answer: Any) -> Dict[str, str]:
        d = {
            'sentence': self.insert_brackets(sample),
            'answer': "是" if answer else "否"
        }
        return d

    def insert_str(self, text, indices, substrs):
        indices = [0] + indices + [None]
        substrs = substrs + ['']
        n = len(indices)
        paired_indices = [(indices[i], indices[i+1]) for i in range(n-1)]
        v = [text[p:q]+c for (p, q), c in zip(paired_indices, substrs)]
        return ''.join(v)

    def insert_brackets(self, d):
        text = d['text']
        s1 = d['target']['span1_index']
        e1 = s1 + len(d['target']['span1_text'])
        s2 = d['target']['span2_index']
        e2 = s2 + len(d['target']['span2_text'])
        if s1 < s2:
            indices = [s1, e1, s2, e2]
            substrs = ['[[', ']]', '<<', '>>']
        else:
            indices = [s2, e2, s1, e1]
            substrs = ['<<', '>>', '[[', ']]']
        return self.insert_str(text, indices, substrs)

    def validate(self, llm_answer: str, std_answer: Any) -> ScoreType:
        idx = llm_answer.find('答案：')
        if idx > -1:
            llm_answer = llm_answer[idx+3:]
        llm_answer = clear_punctuations(llm_answer)
        std_answer = '是' if std_answer else "否"
        return len(llm_answer) > 0 and llm_answer[0] == std_answer
