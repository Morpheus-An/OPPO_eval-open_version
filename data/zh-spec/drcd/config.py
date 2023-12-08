from commons.common_import import *
global ParentDataset
EvalFuncDict.clear()


TaskDescription = """\
請你仔細閱讀以下文章，並且回答文章後的問題。你的回答應該是原文中的一個文字片段，併且盡可能的簡短（比如使用一兩個詞匯，或者一個短片段）。
"""
QuestionPrompt = """\
文章：
{title}  {context}
問題:
{question}
答案：
{answer}
"""


class Dataset(ParentDataset):
    Description = {
        "name": "DRCD",
    }

    _context_list = []

    def analyse_file(self, fp: TextIO, sample_num: Optional[int] = None) -> List[Tuple[Any, Any]]:
        """
        :param fp: 带读取数据集的指针,以文本模式打开,utf-8编码
        :param sample_num: 希望从数据集中读取样例数目,None表示全部读取
        :return: 格式化后的样例列表。每个样例分为两部分,一部分用来生成prompt,一部分为标准答案.每部分具体格式不限制。
        """
        if sample_num is None:
            sample_num = Metric.HUGE_POSITIVE
        dataset = []
        for paragraphs in json.load(fp)["data"][:sample_num]:
            title = paragraphs['title']
            for paragraph in paragraphs['paragraphs']:
                context = paragraph['context']
                context_id = len(self._context_list)
                self._context_list.append(context)
                for question_answer_list in paragraph['qas']:
                    question = question_answer_list['question']
                    answers = list(set([answer['text'] for answer in question_answer_list['answers']]))
                    dataset.append(((title, context_id, question), answers))
        return dataset

    def fill_prompt(self, sample: Any, answer: Any) -> Dict[str, str]:
        """
        :param sample: 一个样例
        :param answer: 该样例对应的答案
        :return: 文件开头Prompt模板中有一些未填充的变量，这里需返回一个字典，以这些变量名称为键，以该样例的信息为值
        """
        title, context_id, question = sample
        return {
            'title': title,
            'context': self._context_list[context_id],
            'question': question,
            'answer': random.choice(answer)
        }

    def validate(self, llm_answer, std_answer) -> Any:
        """
        :param llm_answer: 大模型给出的答案
        :param std_answer: 标准答案
        :return: 对模型给出的答案的评分，一般为布尔值或分数，也可以为其他
        """
        scorer = Metric()
        llm_answer = llm_answer.strip()
        scorer.disable_chinese_cut() # jieba无法实现繁体字分词
        score = max([scorer.Rouge_L(llm_answer, answer) for answer in std_answer])
        self.info = None
        return score

