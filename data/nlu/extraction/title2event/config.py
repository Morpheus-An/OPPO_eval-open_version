from scipy.optimize import linear_sum_assignment
from commons.common_import import *
global ParentDataset
EvalFuncDict.clear()


TaskDescription = """\
根据新闻标题进行事件抽取。\
每个事件是指一个形如“主语|谓语|宾语”的三元组，描述一个新闻事件。\
其中，谓语或宾语可能空缺。\
一个新闻标题中可能抽取出多个事件三元组。\
输出时，请你每行输出一个事件。如果只有一个事件，请类似这样输出：
主语|谓语|宾语
如果给定的句子中能够抽取出多个事件三元组，则类似这样输出：
主语1|谓语1|宾语1
主语2|谓语2|宾语2
...
"""
QuestionPrompt = """\
句子：
{sentence}
答案：
{answer}
"""


class Dataset(ParentDataset):
    Description = {
        "name": "Title2Event"
    }

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
            title = line['title']
            event_info = [eval(f'[{event[f"event{i+1}_triple"]}]')
                          for i, event in enumerate(line['event_info'])]
            dataset.append(((title, ), event_info))
        return dataset

    def fill_prompt(self, sample: Any, answer: Any) -> Dict[str, str]:
        """
        :param sample: 一个样例
        :param answer: 该样例对应的答案
        :return: 文件开头Prompt模板中有一些未填充的变量，这里需返回一个字典，以这些变量名称为键，以该样例的信息为值
        """
        ans = '\n'.join([','.join(i) for i in answer])
        return {
            'sentence': sample,
            'answer': ans
        }

    def validate(self, llm_answer: str, std_answer: Any) -> ScoreType:
        """
        :param llm_answer: 大模型给出的答案
        :param std_answer: 标准答案
        :return: 对模型给出的答案的评分，一般为布尔值或分数，也可以为其他
        注：该函数仅用于评判llm答案的准确性，其他指标请在 _eval_func_list 中实现
        """
        llm_answer = clean_newlines(llm_answer)
        extracted_tuples = []
        for line in llm_answer.split('\n'):
            elements = line.split('|')
            n = len(elements)
            if n > 3:
                elements = elements[:3]
            elif n < 3:
                elements = elements+['']*(3-n)
            extracted_tuples.append(elements)

        scorer = Metric()
        return scorer.Extraction_tri_eval(extracted_tuples, std_answer)

