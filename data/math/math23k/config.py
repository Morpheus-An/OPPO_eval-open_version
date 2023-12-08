from nltk.util import pr
from commons.common_import import *
global ParentDataset
EvalFuncDict.clear()


TaskDescription = """\
请你求解数学应用题。请你通过输出“答案：<你得到的最终答案>”的方式给出答案。\
比如，如果答案是99，请输出“答案：99”\
请你直接给出答案，不要输出任何多余字符
"""
QuestionPrompt = """\
问题：{question}
{answer}
"""


class Dataset(ParentDataset):
    Description = {
        "name": "math23k",
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
            dataset.append(
                ((line['original_text'], line['equation']), line['ans'])
            )
        return dataset

    def fill_prompt(self, sample: Any, answer: Any) -> Dict[str, str]:
        """
        :param sample: 一个样例
        :param answer: 该样例对应的答案
        :return: 文件开头Prompt模板中有一些未填充的变量，这里需返回一个字典，以这些变量名称为键，以该样例的信息为值
        """
        question, equation = sample
        return {
            'question': question,
            'answer': f'问题过程：{equation}\n答案：{answer}'
        }

    def validate(self, llm_answer: str, std_answer: Any) -> Any:
        """
        :param llm_answer: 大模型给出的答案
        :param std_answer: 标准答案
        :return: 对模型给出的答案的评分，一般为布尔值或分数，也可以为其他
        """
        idx = llm_answer.find('答案：')
        if idx != -1:
            seclected_llm_answer = llm_answer[idx+3:]
        else:
            seclected_llm_answer = llm_answer
        idx = seclected_llm_answer.find(std_answer)
        if idx == -1:
            score = 0
        else:
            ans = (' ' + seclected_llm_answer +
                   ' ')[idx: idx + len(std_answer) + 2]
            if ans[0] not in '-0123456789' and ans[-1] not in '0123456789.':
                score = True
            else:
                score = False
        self.info = None
        return score
