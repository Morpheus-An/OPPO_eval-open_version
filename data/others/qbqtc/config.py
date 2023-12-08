from commons.common_import import *
global ParentDataset
EvalFuncDict.clear()

TaskDescription = """\
你会收到两个句子作为输入，请你给出这两个句子的相关性。
你只能输出0,1,2作为回答，数字越大相关性越高：
其中0表示相关程度差，1表示有一定相关性，2表示非常相关。
"""
QuestionPrompt = """\
现有两个句子如下:
句子1：
"{query}"
句子2：
"{title}"
请给出你回答，你只需要输出数字0或1或2即可，不要输出其他的信息!
答案：
{answer}
"""
# dsjflaladsfjljkfd

class Dataset(ParentDataset):
    Description = {
        "name": "QBQTC",
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
            dataset.append(((line['query'], line['title']), line['label']))
        return dataset

    def fill_prompt(self, sample: Any, answer: Any) -> Dict[str, str]:
        """
        :param sample: 一个样例
        :param answer: 该样例对应的答案
        :return: 文件开头Prompt模板中有一些未填充的变量，这里需返回一个字典，以这些变量名称为键，以该样例的信息为值
        """
        query, title = sample
        return {
            'query': query,
            'title': title,
            'answer':answer,
        }

    def validate(self, llm_answer: str, std_answer: Any) -> Any:
        """
        :param llm_answer: 大模型给出的答案
        :param std_answer: 标准答案
        :return: 对模型给出的答案的评分，一般为布尔值或分数，也可以为其他
        """
        # 将答案转换成 one-hot形式
        llm_answer_vector = [llm_answer.find(c) != -1 for c in '012']
        std_answer = int(std_answer)
        if sum(llm_answer_vector) != 1:
            clear_llm_answer = -1
        else:
            clear_llm_answer = llm_answer_vector.index(True)
        self.info = {"F1 Score": (clear_llm_answer, std_answer)}
        return clear_llm_answer == std_answer

    @EvalFunc("F1 Score")
    def score(self, llm_answer_list: List[Any], std_answer_list: List[Any]) -> Union[ScoreType, str]:
        category_list = [0, 1, 2]
        tp_tn_fp_dict = {category: [0, 0, 0] for category in category_list}
        tp_tn_fp_dict |= {-1: [0, 0, 0]}
        for llm_answer, std_answer in zip(llm_answer_list, std_answer_list):
            if llm_answer == std_answer:
                tp_tn_fp_dict[llm_answer][0] += 1
            else:
                tp_tn_fp_dict[llm_answer][2] += 1
                tp_tn_fp_dict[std_answer][1] += 1
        total_f1 = 0
        total_tp, total_tn, total_fp = 0, 0, 0
        ep = 1e-9
        del tp_tn_fp_dict[-1]
        for category in tp_tn_fp_dict:
            tp, tn, fp = tp_tn_fp_dict[category]
            p = (tp + ep) / (tp + fp + ep)
            r = (tp + ep) / (tp + tn + ep)
            total_f1 += 2 * p * r / (p + r)
            total_tp += tp
            total_tn += tn
            total_fp += fp
        p = (total_tp + ep) / (total_tp + total_fp + ep)
        r = (total_tp + ep) / (total_tp + total_tn + ep)
        micro_f1 = 2 * p * r / (p + r)
        macro_f1 = total_f1 / len(category_list)
        return {'macro': macro_f1, 'micro': micro_f1}
