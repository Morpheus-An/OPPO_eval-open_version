from commons.common_import import *
from utils.Metrices.metrics import Metric
global ParentDataset
EvalFuncDict.clear()


TaskDescription = """\
请你以序列标注的形式完成命名实体识别任务, 从句子中识别人名，地名和机构团体。\
你需要在句子中将识别出的实体(可以是词语、句子等)用'[[]]'括起来，并紧接其后，用'(())'输出你识别出的实体类别。\
比如，“国家主席习近平在北京出席重要会议”可以识别为“国家主席[[习近平]]((人名))在[[北京]]((地名))出席重要会议”。\
如果句子中没有属于"人名，地名，机构团体"三者之一的实体，你只需要输出原句即可。
请你直接输出标注结果，不要输出其他多余字符。
"""
# 以下是几个例子以及其解释：
# ------------------
# 输入：国家主席习近平在北京出席重要会议。
# 输出：国家主席[[习近平]]((人名))在[[北京]]((地名))出席重要会议。
# 解释："习近平"是人名；"北京"是地名
# ------------------
# 输入：联合国安理会12月1日通过两项决议,解除对索马里联邦政府的武器禁运,同时延长针对索马里“青年党”的制裁措施
# 输出：[[联合国安理会]]((机构团体))12月1日通过两项决议,解除对[[索马里联邦政府]]((机构团体))的武器禁运,同时延长针对[[索马里“青年党”]]((机构团体))的制裁措施
# 解释："联合国安理会"、"索马里联邦政府"和"索马里“青年党”"都是机构团体
# ------------------
# 输入：秋天的树叶都变红了
# 输出：秋天的树叶都变红了
# 解释：这个句子中没有属于"人名，地名，机构团体"三者之一的实体，因此返回原句。
# ------------------
# 请注意"解释"只是为了让你明白任务的含义。你的回答中不必包含解释。
#
QuestionPrompt = """\
现有句子如下：
{sentence}
答案：
{answer}
"""


class Dataset(ParentDataset):
    Description = {
        "name": "MSRA - NER"
    }

    def analyse_file(self, fp: TextIO, sample_num: Optional[int] = None) -> List[Tuple[Any, Any]]:
        """
        :param fp: 带读取数据集的指针，以文本模式打开，utf-8编码
        :param sample_num: 希望从数据集中读取样例数目，None表示全部读取
        :return: 格式化后的样例列表。每个样例分为两部分，一部分用来生成prompt，一部分为标准答案，每部分具体格式不限制。
        """

        """
        由于llm不稳定的输出：

        llm_ans : 最近一段时间，台湾方面一些重要人士一再明确地表示可以谈政治问题，
        “其中最重要、最优先的课题是结束敌对状态、签署和平协定”，[[台湾]]((地名))方面 。

        std_ans : 最近一段时间，[[台湾]]((地名))方面一些重要人士一再明确地表示可以谈政治问题，
        “其中最重要、最优先的课题是结束敌对状态、签署和平协定”。

        目前没有想到合适的避免措施。因此我放弃了将句子转化为0~3(/o,/nt,/nr,/ns)表示的序列来使用sklearn标准的f1
        求score(它们甚至可能不等长)，而是使用nltk标准的F1来计算。

        ## nltk标准的F1：
        Given a set of reference values and a set of test values, return
        the f-measure of the test values, when compared against the
        reference values.  The f-measure is the harmonic mean of the
        ``precision`` and ``recall``, weighted by ``alpha``.  In particular,
        given the precision *p* and recall *r* defined by:

        - *p* = card(``reference`` intersection ``test``)/card(``test``)
        - *r* = card(``reference`` intersection ``test``)/card(``reference``)

        The f-measure is:

        - *1/(alpha/p + (1-alpha)/r)*

        具体方法:
        llm_ans : 最近一段时间，台湾方面一些重要人士一再明确地表示可以谈政治问题，
        “其中最重要、最优先的课题是结束敌对状态、签署和平协定”，[[台湾]]((地名))方面 。

        从llm的回答中抽取出-> ['台湾是地名']

        std_ans : 最近一段时间，[[台湾]]((地名))方面一些重要人士一再明确地表示可以谈政治问题，
        “其中最重要、最优先的课题是结束敌对状态、签署和平协定”。

        -> ['台湾是地名']

        scorer.F1(llm_ans, std_ans, ordered = False) = 1.0
        """
        if sample_num == None:
            sample_num = 99999999
        dataset = []
        while line := fp.readline():
            if len(dataset) > sample_num:
                break
            answer = line
            # padding 到 2
            answer = answer.replace('/o', '/o ')
            index_list = []
            for mark in ['/o ', '/nr', '/ns', '/nt']:
                next_index = 0
                while True:
                    next_index = answer.find(mark, next_index)
                    if next_index == -1:
                        break
                    index_list.append(next_index)
                    next_index += 3
            index_list.sort()

            start = 0
            ans_list = []
            original_answer = answer[:]
            mapping = {'/nr': '人名', "/ns": '地名', '/nt': '机构团体'}
            for cur in index_list:
                if start >= len(answer):
                    break
                mark = answer[cur:cur+3]
                if mark != '/o ':
                    # 只考虑 '/nr', '/ns', '/nt'
                    tar = answer[start:cur].strip()
                    # 不破坏空格数
                    leftspaces, rightspaces = 0, 0
                    for i in answer[start:cur]:
                        if i == tar[0]:
                            break
                        leftspaces += 1
                    for i in answer[start:cur][::-1]:
                        if i == tar[-1]:
                            break
                        rightspaces += 1
                    leftspaces, rightspaces = leftspaces * ' ', rightspaces * ' '
                    # 获取答案信息
                    ans_list.append(tar + f"是{mapping[mark]}")
                    # 获取答案表示
                    appdx = f"[[{tar}]](({mapping[mark]}))"
                    answer = answer[:start] + leftspaces + \
                        appdx + rightspaces + answer[cur:]
                    delta_len = len(appdx) + len(leftspaces +
                                                 rightspaces) - (cur - start)
                    for i in range(len(index_list)):
                        index_list[i] += delta_len
                    start = cur + delta_len + 3
                else:
                    start = cur + 3

            pure = original_answer.replace(
                '/o  ', '').replace('/nr ', '').replace('/ns ', '').replace('/nt ', '')
            answer = answer.replace(
                '/o  ', '').replace('/nr ', '').replace('/ns ', '').replace('/nt ', '')
            # print(f"pure = {pure}, real_ans = {answer}, anslist = {ans_list}")
            dataset.append((pure, (answer.strip(), ans_list)))
        return dataset

    def fill_prompt(self, sample: Any, answer: Any) -> Dict[str, str]:
        """
        :param sample: 一个样例
        :param answer: 该样例对应的答案
        :return: 文件开头Prompt模板中有一些未填充的变量，这里需返回一个字典，以这些变量名称为键，以该样例的信息为值
        """
        return {
            'sentence': sample,
            'answer': answer[0]
        }

    def validate(self, llm_answer: str, std_answer: Any) -> ScoreType:
        """
        :param llm_answer: 大模型给出的答案
        :param std_answer: 标准答案
        :return: 对模型给出的答案的评分，一般为布尔值或分数，也可以为其他
        注：该函数仅用于评判llm答案的准确性，其他指标请在 _eval_func_list 中实现
        """
        if not std_answer[1]:
            # 没有实体
            if llm_answer.strip() == std_answer[0]:
                return 1
            else:
                return 0

        index_list = []
        next_index = 0
        while True:
            next_index = llm_answer.find("[[", next_index)
            if next_index == -1:
                break
            else:
                index_list.append(next_index)
                next_index += 2

        llm_list = []
        for point in index_list:
            # 匹配 prefix : [[XXX]]
            start = point + 2
            end = start
            while end < len(llm_answer) - 1:
                if llm_answer[end:end+2] == ']]':
                    break
                else:
                    end += 1
            if end >= len(llm_answer) - 4 or end == start:
                # llm 输出异常
                return 0
            if llm_answer[end+2:end+4] != '((':
                # 没有后缀
                continue
            prefix = llm_answer[start:end]

            # 匹配 suffix : ((XXX))
            start = end + 4
            end = start
            while end < len(llm_answer) - 1:
                if llm_answer[end:end+2] == '))':
                    break
                else:
                    end += 1
            if end >= len(llm_answer) or end == start:
                # llm 输出异常
                return 0
            suffix = llm_answer[start:end]
            llm_list.append(prefix + "是" + suffix)
        scorer = Metric()
        return scorer.F1(llm_list, std_answer[1], ordered=False)

