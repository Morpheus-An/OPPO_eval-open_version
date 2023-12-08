from commons.common_import import *
global ParentDataset
EvalFuncDict.clear()


TaskDescription = """\
请你完成成语填空任务。\
在所给的文段中，缺失的文字被标记为#idiom#，\
请你从待选的成语中选择 1 个最合适的成语填入，使得文段句意通顺。\
文段可能会有多处缺失，每处都会有若干个待选的成语，请你输出时按顺序每行输出一个。
"""
QuestionPrompt = """\
[文段]：
{passage}
[选项]：
{candidates}
[答案]：
{answer}
"""

temp = "（请按顺序输出你选择的成语，每行一个）"


class Dataset(ParentDataset):
    Description = {
        "name": "CHiD",
    }

    _passages = []

    def analyse_file(self, fp: TextIO, sample_num: Optional[int] = None) -> List[Tuple[Any, Any]]:
        """
        :param fp: 带读取数据集的指针，以文本模式打开，utf-8编码
        :param sample_num: 希望从数据集中读取样例数目，None表示全部读取
        :return: 格式化后的样例列表。每个样例分为两部分，一部分用来生成prompt，一部分为标准答案，每部分具体格式不限制。
        """
        if sample_num is None:
            sample_num = Metric.HUGE_POSITIVE
        dataset = []
        while line := fp.readline():
            if len(dataset) >= sample_num:
                break
            line = eval(line)
            real_count = line['realCount']
            passage = ''.join([part + blank
                               for part, blank in zip(line['content'].split('#idiom#'),
                                                      [f"#idiom#[{i}]" for i in range(real_count)] + [''])])
            dataset.append(
                ((passage, line['candidates'], real_count), line['groundTruth']))
        return dataset

    def fill_prompt(self, sample: Any, answer: Any) -> Dict[str, str]:
        """
        :param sample: 一个样例
        :param answer: 该样例对应的答案
        :return: 文件开头Prompt模板中有一些未填充的变量，这里需返回一个字典，以这些变量名称为键，以该样例的信息为值
        """
        passage, candidates_list, real_count = sample
        return {
            'passage': passage,
            'candidates': '\n'.join([f"#idiom#[{i}]处的候选成语：\n" + ', '.join(candidates)
                                     for i, candidates in enumerate(candidates_list)]),
            'answer': '\n'.join(answer),
        }

    def validate(self, llm_answer: str, std_answer: Any) -> Any:
        """
        :param llm_answer: 大模型给出的答案
        :param std_answer: 标准答案
        :return: 对模型给出的答案的评分，一般为布尔值或分数，也可以为其他
        """
        pos_list = [-4]
        for idiom in std_answer:
            pos = llm_answer.find(idiom, pos_list[-1] + 4)
            if pos != -1:
                pos_list.append(pos)
        score = (len(pos_list) - 1) / len(std_answer)
        self.info = None
        return score
