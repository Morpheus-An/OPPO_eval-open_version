from commons.common_import import *
global ParentDataset
EvalFuncDict.clear()


TaskDescription = """\
仔细阅读以下论文摘要，并根据摘要生成中文论文标题。
"""
QuestionPrompt = """\
摘要：{content}
标题：{answer}
"""


class Dataset(ParentDataset):
    Description = {
        "name": "CSL - ts"
    }

    def analyse_file(self, fp: TextIO, sample_num: Optional[int] = None) -> List[Tuple[Any, Any]]:
        """
        :param fp: 带读取数据集的指针，以文本模式打开，utf-8编码
        :param sample_num: 希望从数据集中读取样例数目，None表示全部读取
        :return: 格式化后的样例列表。每个样例分为两部分，一部分用来生成prompt，一部分为标准答案，每部分具体格式不限制。
        """
        if sample_num == None:
            sample_num = Metric.HUGE_POSITIVE
        TRUNCATED_LENGTH = 600
        print(f"{BOLD_GREEN} csl-ts 数据集的文章统一截断为{BOLD_RED}{TRUNCATED_LENGTH}{BOLD_GREEN}，以避免长度超限{RESET_COLOR}")
        dataset = []
        while line := fp.readline():
            if len(dataset) > sample_num:
                break
            _, content, title = line.strip().split('\t')
            dataset.append(((content[:TRUNCATED_LENGTH], ), title))
        return dataset

    def fill_prompt(self, sample: Any, answer: Any) -> Dict[str, str]:
        """
        :param sample: 一个样例
        :param answer: 该样例对应的答案
        :return: 文件开头Prompt模板中有一些未填充的变量，这里需返回一个字典，以这些变量名称为键，以该样例的信息为值
        """
        content, *_ = sample
        return {
            'content': content,
            'answer': answer
        }

    def validate(self, llm_answer: str, std_answer: Any) -> ScoreType:
        """
        :param llm_answer: 大模型给出的答案
        :param std_answer: 标准答案
        :return: 对模型给出的答案的评分，一般为布尔值或分数，也可以为其他
        注：该函数仅用于评判llm答案的准确性，其他指标请使用 EvalFunc 装饰器添加（参见下面的函数）。
        """
        scorer = Metric()
        # res = scorer.F1(llm_answer,std_answer, smooth_threshold=0.5)
        res = scorer.Rouge_L(llm_answer, std_answer)
        return res
