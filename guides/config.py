from commons.common_import import *
global ParentDataset
EvalFuncDict.clear()


TaskDescription = """\
...任务描述...
"""  # 纯文本，不应当有任何变量
QuestionPrompt = """\
...问题描述...
{answer}
"""  # 可以包含多个变量，最后一个变量必须是 answer


class Dataset(ParentDataset):
    Description = {
        "name": ...,  # 数据集名称, 被用在 defaults.py文件中
    }

    def analyse_file(self, fp: TextIO, sample_num: Optional[int] = None) -> List[Tuple[Any, Any]]:
        """
        :param fp: 带读取数据集的指针，以文本模式打开，utf-8编码,
            相当于 fp = open(..., "r", encoding="utf-8")
        :param sample_num: 希望从数据集中读取样例数目，None表示全部读取
        :return: 格式化后的样例列表, 列表中的每个元素都是一个二元组，
            第一部分用来生成prompt，第二部分为标准答案，每部分具体格式不限制。
        """

        ...

        # 这里可以打印几个样例来debug
        pass

    def fill_prompt(self, sample: Any, answer: Any) -> Dict[str, str]:
        """
        :param sample: 一个样例
        :param answer: 该样例对应的答案
        :return: 文件开头Prompt模板 `QuestionPrompt` 中有一些未填充的变量，
            这里需返回一个字典，以这些变量名称为键，以该样例的信息为值
        """

        ...

        # 这里可以打印几个样例来debug
        pass

    def validate(self, llm_answer: str, std_answer: Any) -> ScoreType:
        """
        :param llm_answer: 大模型的输出
        :param std_answer: 标准答案
        :return: 对模型输出的评分
        注：该函数仅用于评判llm在这一个样例的输出的评分，其他指标请使用 EvalFunc 装饰器添加（参见下面的函数）。
        """
        self.info = {
            "score_func1": (Any, Any),
            "score_func2": (Any, Any, Any),
            "...        ": ...
        }
        # self.info 需要传回一些可能用于其他指标计算的信息，根据数据集题型不同而有差异
        # self.info 的每一个键必须对应一个 EvalFunc 的名称，其包含的内容也必须是该 EvalFunc 的参数（不包括 self 参数）。
        pass

    # 你可以不定义任何额外的指标计算函数，那么删除以下部分即可
    # 你也可以继续添加其他指标的计算函数，那么只需复制以下部分再修改即可
    # 如果你增加了新的指标计算函数，EvalFunc装饰器中的名称请不要重复。
    # 或者，你可以使用 @EvalFunc() 或 @EvalFunc("")，这种情况下将使用函数的原名称
    @EvalFunc("score_func1")
    def score(self, para_1: List[Any], para_2: List[Any]) -> Union[ScoreType, str]:
        pass

    @EvalFunc("score_func2")
    def score(self, para_1: List[Any], para_2: List[Any], para_3: List[Any]) -> Union[ScoreType, str]:
        pass

    ...

