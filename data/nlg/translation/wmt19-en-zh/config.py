from commons.common_import import *
global ParentDataset
EvalFuncDict.clear()


TaskDescription = """\
请你把下面的英文句子翻译成中文。翻译的要求是信、达、雅，你可以对语序、用词等方面进行适当的调整。
"""
QuestionPrompt = """\
英文：
{origin}
中文:
{answer}
"""


class Dataset(ParentDataset):
    Description = {
        "name": "WMT19 - en2zh",
    }

    def analyse_file(self, fp: TextIO, sample_num: Optional[int] = None) -> List[Tuple[Any, Any]]:
        """
        :param fp: 带读取数据集的指针，以文本模式打开，utf-8编码
        :param sample_num: 希望从数据集中读取样例数目，None表示全部读取
        :return: 格式化后的样例列表。每个样例分为两部分，一部分用来生成prompt，一部分为标准答案，每部分具体格式不限制。
        """
        dataset = []
        data_list = json.load(fp)
        if sample_num is None:
            sample_num = Metric.HUGE_POSITIVE
        for sample in data_list[:sample_num]:
            en = sample['src']
            ch = sample['ref']
            dataset.append(((en, ), ch))
        return dataset

    def fill_prompt(self, sample: Any, answer: Any) -> Dict[str, str]:
        """
        :param sample: 一个样例
        :param answer: 该样例对应的答案
        :return: 文件开头Prompt模板中有一些未填充的变量，这里需返回一个字典，以这些变量名称为键，以该样例的信息为值
        """
        en, = sample
        return{
            'origin': en,
            'answer': answer,
        }

    def validate(self, llm_answer: str, std_answer: Any) -> ScoreType:
        """
        :param llm_answer: 大模型给出的答案
        :param std_answer: 标准答案
        :return: 对模型给出的答案的评分，一般为布尔值或分数，也可以为其他
        注：该函数仅用于评判llm答案的准确性，其他指标请使用 EvalFunc 装饰器添加（参见下面的函数）。
        """
        scorer = Metric()
        return scorer.BLEU_sentence_level(llm_answer,std_answer)
