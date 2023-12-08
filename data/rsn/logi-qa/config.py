from commons.common_import import *
global ParentDataset
EvalFuncDict.clear()


TaskDescription = """\
请阅读以下描述，并回答描述后面的问题：
"""
QuestionPrompt = """
{description}
问题: {question}
（请选择正确答案，你的回答一定要完整地包括正确答案的序号和内容）
{choices} 
答案：
{answer}
"""

class Dataset(ParentDataset):
    Description = {
        "name": "LogiQA",
    }

    _passage_list = []

    def analyse_file(self, fp: TextIO, sample_num: Optional[int] = None) -> List[Tuple[Any, Any]]:
        """
        :param fp: 带读取数据集的指针，以文本模式打开，utf-8编码
        :param sample_num: 希望从数据集中读取样例数目，None表示全部读取
        :return: 格式化后的样例列表。每个样例分为两部分，一部分用来生成prompt，一部分为标准答案，每部分具体格式不限制。
        """
        if sample_num is None:
            sample_num = Metric.HUGE_POSITIVE
        dataset = []
        ans_list = ['a\n', 'b\n', 'c\n', 'd\n']
        while line := fp.readline():
            if line in ans_list:
                description = fp.readline()[:-1]
                question = fp.readline()[:-1]
                choices = [fp.readline()[2:-1].strip() for _ in range(4)]
                answer = choices[ans_list.index(line)]
                dataset.append(((description, question, choices), answer))

        return dataset

    def fill_prompt(self, sample: Any, answer: Any) -> Dict[str, str]:
        """
        :param sample: 一个样例
        :param answer: 该样例对应的答案
        :return: 文件开头Prompt模板中有一些未填充的变量，这里需返回一个字典，以这些变量名称为键，以该样例的信息为值
        """
        letter_list = "ABCDE"
        description, question, choices = sample
        answer_idx = -1
        for i, choice in enumerate(choices):
            if choice == answer:
                answer_idx = i
                break
        return {
            'description': description,
            'question': question,
            'choices': '\n'.join([f"{letter}.{choice}" for letter, choice in zip(letter_list, choices)]),
            'answer': f'{letter_list[answer_idx]}.{answer}'
        }
        # return f"请阅读以下文章，并回答文章后面的问题：\n{self._passage_list[passage_idx]}\n" +\
        #        f"问题: {question}\n（请选择正确答案，你的回答一定要完整地包括正确答案的序号和内容）\n" +\
        #        '\n'.join([f"{letter}.{choice}" for letter, choice in zip("ABCDE", choice_list)])

    def validate(self, llm_answer: str, std_answer: Any) -> Any:
        """
        :param llm_answer: 大模型给出的答案
        :param std_answer: 标准答案
        :return: 对模型给出的答案的评分，一般为布尔值或分数，也可以为其他
        """
        for line in llm_answer.split('\n')[::-1]:
            for _letter in 'ABCDE':
                pos = line.find(_letter + '.')
                if pos != -1:
                    score = line[pos + 2:].strip()[:len(std_answer)] == std_answer
                    break
            else:
                continue
            break
        else:
            score = llm_answer.find(std_answer) != -1
        self.info = None
        return score
