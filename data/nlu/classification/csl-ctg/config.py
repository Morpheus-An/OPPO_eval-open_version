from tqdm import std
from commons.common_import import *
global ParentDataset
EvalFuncDict.clear()


TaskDescription = """\
请你完成论文学科分类任务。\
给定一个论文标题，\
请你对其所属学科领域进行分类。
一共有13个类别，分别是：
历史学，管理学，教育学，心理学，医学，工学，经济学，法学，军事学，农学，文学，哲学，艺术。
请你直接输出13个类别中的一个，不要输出多余文字。
"""
QuestionPrompt = """\
内容：{content}
类别：{answer}
"""


class Dataset(ParentDataset):
    Description = {
        "name": "CSL - ctg"
    }
    available_classes = [
        '历史学', '管理学', '教育学', '心理学', '医学', '工学', '经济学', '法学', '军事学', '农学', '文学', '哲学', '艺术'
    ]

    def analyse_file(self, fp: TextIO, sample_num: Optional[int] = None) -> List[Tuple[Any, Any]]:
        """
        :param fp: 带读取数据集的指针，以文本模式打开，utf-8编码
        :param sample_num: 希望从数据集中读取样例数目，None表示全部读取
        :return: 格式化后的样例列表。每个样例分为两部分，一部分用来生成prompt，一部分为标准答案，每部分具体格式不限制。
        """
        if sample_num == None:
            sample_num = Metric.HUGE_POSITIVE
        dataset = []
        while line := fp.readline():
            if len(dataset) > sample_num:
                break
            _, content, keywords = line.strip().split('\t')
            dataset.append(((content, ), keywords))
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
        llm_answer = clean_newlines(llm_answer)
        lines = llm_answer.split('\n')
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
