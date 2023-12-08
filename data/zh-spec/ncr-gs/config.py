from commons.common_import import *
global ParentDataset
EvalFuncDict.clear()


TaskDescription = """\
请阅读以下文章，并回答文章后的问题，问题有A,B,C,D四个选项\
，其中只有一个是正确答案，你只需要给出A,B,C,D中的一个字母，表示你认为正确的选项即可，不要输出其他信息。
"""

QuestionPrompt = """\
文章：
{content}
问题：{question}
选项：{choices}
答案：{answer}
"""


class Dataset(ParentDataset):
    Description = {
        'name': 'NCR - gs'
    }
    available_classes = list("ABCD")

    def analyse_file(self, fp: TextIO, sample_num: Optional[int] = None) -> List[Tuple[Any, Any]]:
        """
        :param fp: 带读取数据集的指针，以文本模式打开，utf-8编码
        :param sample_num: 希望从数据集中读取样例数目，None表示全部读取
        :return: 格式化后的样例列表。每个样例分为两部分，一部分用来生成prompt，一部分为标准答案，每部分具体格式不限制。
        """
        original_data = json.load(fp)
        data = []
        if sample_num is None:
            sample_num = float('inf')
        for i in range(min(sample_num, len(original_data))):
            content = original_data[i]['Content']
            questions = original_data[i]['Questions']
            for question in questions:
                q = question['Question']
                c = '\n'.join(question['Choices'])
                a = question['Answer']
                data.append(
                    ({'content': content, 'question': q, 'choices': c}, a))
        return data

    def fill_prompt(self, sample: Any, answer: Any) -> Dict[str, str]:
        """
        :param sample: 一个样例
        :param answer: 该样例对应的答案
        :return: 文件开头Prompt模板中有一些未填充的变量，这里需返回一个字典，以这些变量名称为键，以该样例的信息为值
        """
        return {
            'content': sample['content'],
            'question': sample['question'],
            'choices': sample['choices'],
            'answer': answer
        }

    def validate(self, llm_answer: str, std_answer: Any) -> ScoreType:
        """
        :param llm_answer: 大模型给出的答案
        :param std_answer: 标准答案
        :return: 对模型给出的答案的评分，一般为布尔值或分数，也可以为其他
        注：该函数仅用于评判llm答案的准确性，其他指标请使用 EvalFunc 装饰器添加（参见下面的函数）。
        """
        llm_answer = clean_newlines(llm_answer)
        lines = llm_answer.upper().split('\n')
        score = 0
        for line in lines[::-1]:
            hits = [(c in line) for c in self.available_classes]
            if sum(hits) == 1:
                idx = hits.index(True)
                ans = self.available_classes[idx]
                score = int(ans == std_answer)
                break
        self.info = None
        return score
