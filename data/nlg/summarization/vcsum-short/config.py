from commons.common_import import *
global ParentDataset
EvalFuncDict.clear()


TaskDescription = """\
请你阅读所给会议记录，并撰写会议摘要。
"""  # 纯文本，不应当有任何变量
QuestionPrompt = """\
会议记录：
{content}
摘要：
{answer}
"""  # 可以包含多个变量，最后一个变量必须是 answer


class Dataset(ParentDataset):
    Description = {
        "name": "VCSum - short"
    }

    def analyse_file(self, fp: TextIO, sample_num: Optional[int] = None) -> List[Tuple[Any, Any]]:
        """
        :param fp: 带读取数据集的指针，以文本模式打开，utf-8编码
        :param sample_num: 希望从数据集中读取样例数目，None表示全部读取
        :return: 格式化后的样例列表。每个样例分为两部分，一部分用来生成prompt，一部分为标准答案，每部分具体格式不限制。
        """
        TRUNCATED_LENGTH = 3200
        # 由于上下文长度限制，这里对于任意长度的长文本，可能需要进行截断。
        # 因此目前该指标的计算结果不准。后续如果使用16K或更长模型，可以修改该常数。
        if sample_num == None:
            sample_num = Metric.HUGE_POSITIVE
        dataset = []
        while line := fp.readline():
            if len(dataset) >= sample_num:
                break
            line = eval(line)
            summary = line['discussion']
            content = line['context']
            speaker = line['speaker']
            full_content = ''.join([f'说话人{speaker_id}号：\n' + '\n'.join(part_content)
                                    for part_content, speaker_id in zip(content, speaker)])[:TRUNCATED_LENGTH]
            dataset.append(((full_content, ), summary))
        return dataset

    def fill_prompt(self, sample: Any, answer: Any) -> Dict[str, str]:
        """
        :param sample: 一个样例
        :param answer: 该样例对应的答案
        :return: 文件开头Prompt模板中有一些未填充的变量，这里需返回一个字典，以这些变量名称为键，以该样例的信息为值
        """
        content, = sample
        return {
            'content': content,
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
        return scorer.Rouge_L(llm_answer, std_answer)

