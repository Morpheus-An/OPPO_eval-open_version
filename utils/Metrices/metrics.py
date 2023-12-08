if __name__ == '__main__':
    from rouge_chinese.rouge import Rouge
    from specialize.extract import extract_tri_eval
else:
    from .rouge_chinese.rouge import Rouge
    from .specialize.extract import extract_tri_eval
import re
import logging
import pandas as pd
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from sklearn.metrics import precision_score, recall_score, accuracy_score
from typing import List, Tuple, Dict, Any
import warnings
warnings.filterwarnings("ignore")


class Metric:
    """
    Metric 类
    - 提供以下函数的接口：
        -  Rouge_1 ~ Rouge_5
        -  Rouge_L
        -  BLEU_sentence_level  支持1(candidate),1(reference)
        -  BLEU_corpus_level    批量的bleu值计算. 支持[M * 1(candidate)],[M * [n * (reference)]]
        -  Accuracy, Precision, Recall, F1
        -  Extraction_tri_eval  用于事件抽取中三元组的评估

    - 提供以下常量定义
        -  HUGE_POSITIVE, HUGE_NEGATIVE
    """
    DEFAULT_CONFIG = {
        'default_rouge_metrics': ["rouge-1", "rouge-2", "rouge-3", "rouge-4", "rouge-5", "rouge-l"],
        'chinese_cut': True,
    }

    HUGE_POSITIVE = 999999999
    HUGE_NEGATIVE = -999999999

    def __init__(self, default_config=DEFAULT_CONFIG):
        self._default_rouge_metrics = default_config['default_rouge_metrics']
        self._chinese_cut = default_config['chinese_cut']
        self._chinese_cut_fn = None
        if self._chinese_cut:
            try:
                import jieba
                self._chinese_cut_fn = jieba.cut
                jieba.setLogLevel(logging.INFO)
            except:
                assert (0), '缺少以下包 : jieba'
        self._default_rouge = Rouge(self._default_rouge_metrics)
        self._default_bleu = corpus_bleu

    def __process_sentence(
        self, s: str,
        language: str = 'ch',
        join: bool = True,
        trace: Dict = None
    ):
        """
        processing sentences
        """
        join_fn = ' '.join if join else lambda x: x
        # cut
        if language == 'ch':
            if not self._chinese_cut:
                s = join_fn(list(s))
            else:
                s = join_fn(self._chinese_cut_fn(s))
        # lower and remove spaces
        s = s.lower() if join else list(
            filter(lambda x: x.strip() != '', [i.lower() for i in s]))
        # tracing bleu weights
        if trace:
            if join:
                tmp = s[:]
                tmp = tmp.split()
            else:
                tmp = s
            trace['bleu_max_grams'] = min(len(tmp), trace['bleu_max_grams'])
            # print(f"s = {s}, max_n_grams = {len(tmp)}")
        return s

    def disable_chinese_cut(self):
        """
        禁用中文分词
        """
        self._chinese_cut = False

    def enable_chinese_cut(self):
        """
        启用中文分词
        """
        self._chinese_cut = True

    def __check_language(self, s1: str, s2: str):
        pattern = re.compile(r'[\u4e00-\u9fff]+')
        res1 = pattern.search(s1)
        res2 = pattern.search(s2)
        return 'ch' if res1 or res2 else 'en'

    def __Rouge_fn(
        self,
        tar: str,
        llm_answer: str,
        std_answer: str,
        return_dict: bool,
        **kargs
    ):
        if not tar in self._default_rouge_metrics:
            print(f'\"{tar}\" 并不在 Metric 的初始化范围内。请修改 DEFAULT_CONFIG')
        llm_answer, std_answer = self.__process_sentence(
            llm_answer), self.__process_sentence(std_answer)
        score_fn = self._default_rouge.get_scores
        score_dict = score_fn(llm_answer, std_answer)[0][tar]
        if return_dict:
            return score_dict
        else:
            if kargs:
                p = score_dict['p']
                r = score_dict['r']
                beta = kargs['beta']
                return (1 + beta**2) * r * p / (r + beta**2 * p + 1e-8)
            else:
                return score_dict['f']

    def Rouge_L(
        self,
        llm_answer: str,
        std_answer: str,
        beta: float = 1.0,
        return_dict: bool = False
    ):
        """
        计算rouge-l指标
        """
        return self.__Rouge_fn('rouge-l', llm_answer, std_answer, return_dict, beta=beta)

    def Rouge_1(
        self,
        llm_answer: str,
        std_answer: str,
        return_dict: bool = False
    ):
        """
        计算rouge-1指标
        """
        return self.__Rouge_fn('rouge-1', llm_answer, std_answer, return_dict)

    def Rouge_2(
        self,
        llm_answer: str,
        std_answer: str,
        return_dict: bool = False
    ):
        """
        计算rouge-2指标
        """
        return self.__Rouge_fn('rouge-2', llm_answer, std_answer, return_dict)

    def Rouge_3(
        self,
        llm_answer: str,
        std_answer: str,
        return_dict: bool = False
    ):
        """
        计算rouge-3指标
        """
        return self.__Rouge_fn('rouge-3', llm_answer, std_answer, return_dict)

    def Rouge_4(
        self,
        llm_answer: str,
        std_answer: str,
        return_dict: bool = False
    ):
        """
        计算rouge-4指标
        """
        return self.__Rouge_fn('rouge-4', llm_answer, std_answer, return_dict)

    def Rouge_5(
        self,
        llm_answer: str,
        std_answer: str,
        return_dict: bool = False
    ):
        """
        计算rouge-5指标
        """
        return self.__Rouge_fn('rouge-5', llm_answer, std_answer, return_dict)

    def BLEU_sentence_level(
        self,
        llm_answer: str,
        std_answer: str,
        smooth: bool = True,
    ) -> float:
        """
        sentence_level_bleu
        ------------------------
        采用 method0 作为 smoothing_method，smooth = False 可能导致score偏小。

        Examples:

        >>> BLEU_sentence_level("你好"，"你好啊", smooth = True)

        """
        return self.BLEU_corpus_level([llm_answer], [[std_answer]], smooth)

    def BLEU_corpus_level(
        self,
        llm_corpus: List[str],
        std_corpus: List[List[str]],
        smooth: bool = True,
    ) -> float:
        """
        corpus_level_bleu
        ------------------------
        采用 method0 作为 smoothing_method，smooth = False 可能导致score偏小。

        Examples:

            >>> BLEU_corpus_level(["candidate1","candidate2"]，
                                  [["ref11", "ref12"],
                                   ["ref21", "ref22"]])

            >>> BLEU_corpus_level(llm_corpus = [ "你好", "动物园"]，
                                  std_corpus = [["你好", "你好呀", "你好啊"],
                                                ["动物森林"]],
                                  smooth = True)

        """

        if smooth:
            smoothing_fn = SmoothingFunction().method0
        else:
            smoothing_fn = None
        trace = {}
        trace['bleu_max_grams'] = self.HUGE_POSITIVE
        translation_corpus = [self.__process_sentence(
            i, join=False, trace=trace) for i in llm_corpus]
        reference_corpus = [[self.__process_sentence(
            i, join=False, trace=trace) for i in j] for j in std_corpus]

        N = trace['bleu_max_grams']
        if N > 4:
            weights = [0.25, 0.25, 0.25, 0.25]
        else:
            weights = [1 / N if i < N else 0 for i in range(4)]

        # print(translation_corpus, reference_corpus, weights)
        score = self._default_bleu(list_of_references=reference_corpus,
                                   hypotheses=translation_corpus,
                                   smoothing_function=smoothing_fn,
                                   weights=weights)
        return score

    def __gin(self, curstr: Any, tarlist: List[Any], smooth_threshould: float) -> bool:
        if curstr in tarlist:
            return True
        if smooth_threshould < 1:
            for i in tarlist:
                if type(curstr) == type(i) == str:
                    similarity = self.Rouge_L(curstr, i)
                    # print(f"r-l sim for {curstr}, {i} is {similarity}")
                    if similarity >= smooth_threshould:
                        return True
        return False

    def __pracf_check(
        self,
        llm_answer: List[Any],
        std_answer: List[Any],
        ordered: bool,
        smooth_threshould: float
    ) -> Tuple[List[Any], List[Any]]:
        assert (0 <= smooth_threshould <= 1), "smooth_threshould 必须位于 0 ~ 1 之间"
        if ordered:
            assert (len(llm_answer) == len(std_answer)), "输入序列长度不一致"
            if smooth_threshould < 1:
                for i in range(len(llm_answer)):
                    if type(llm_answer[i]) == type(std_answer[i]) == str:
                        similarity = self.Rouge_L(llm_answer[i], std_answer[i])
                        if similarity >= smooth_threshould:
                            llm_answer[i] = std_answer[i]
        else:
            llm_answer = list(set(llm_answer))
            std_answer = list(set(std_answer))
        return llm_answer, std_answer

    def Accuracy(
        self,
        llm_answer: List[Any],
        std_answer: List[Any],
        smooth_threshould: float = 1.0
    ):
        """
        ACC 计算
        ---------------
        计算sklearn标准的 ACC 指标。
        ### smooth_threshould : float, default = 1.0, range = 0.0 ~ 1.0
            llm 的字符串输出往往存在偏差，按照exact match可能会出现score多为0的情况。例如
            candidate : ['大模型', '自然语言处理'], ref : ['大语言模型', '自然语言处理技术']。smooth_threshould
            给予了一定容忍阈值。设为1则不进行smooth，否则认为相似度大于阈值的一对字符串exact match.
        """
        if smooth_threshould < 1:
            for i in range(len(llm_answer)):
                if type(llm_answer[i]) == type(std_answer[i]) == str:
                    similarity = self.Rouge_L(llm_answer[i], std_answer[i])
                    if similarity >= smooth_threshould:
                        llm_answer[i] = std_answer[i]
        return accuracy_score(std_answer, llm_answer)

    def Precision(
        self,
        llm_answer: List[Any],
        std_answer: List[Any],
        average: str = 'micro',
        ordered: bool = True,
        smooth_threshould: float = 1.0
    ) -> float:
        """
        P/R/F1 计算
        ---------------
        ### ordered : bool : True | False, default = True
            当 ordered 为 True 时， 计算sklearn标准的 P/R/F1 指标。此时要求 llm_answer 和 std_answer 必须等长。
            适用于分类任务;

            当 ordered 为 False 时，计算nltk标准的 P/R/F1 指标。此时认为 llm_answer 和 std_answer 不计顺序，
            不必等长。若存在重复元素，则默认去重后计算。适用于关键词提取等任务

        ### smooth_threshould : float, default = 1.0, range = 0.0 ~ 1.0
            llm 的字符串输出往往存在偏差，按照exact match可能会出现score多为0的情况。例如
            candidate : ['大模型', '自然语言处理'], ref : ['大语言模型', '自然语言处理技术']。smooth_threshould
            给予了一定容忍阈值。设为1则不进行smooth，否则认为相似度大于阈值的一对字符串exact match.
        """

        llm_answer, std_answer = self.__pracf_check(
            llm_answer, std_answer, ordered, smooth_threshould)
        if ordered:
            return precision_score(std_answer, llm_answer, average=average)
        count = 0
        for i in range(len(llm_answer)):
            if self.__gin(llm_answer[i], std_answer, smooth_threshould=smooth_threshould):
                count += 1
        return count / (len(llm_answer) + 1e-8)

    def Recall(
        self,
        llm_answer: List[Any],
        std_answer: List[Any],
        average: str = 'micro',
        ordered: bool = True,
        smooth_threshould: bool = 1.0
    ) -> float:
        """
        P/R/F1 计算
        ---------------
        ### ordered : bool : True | False, default = True
            当 ordered 为 True 时， 计算sklearn标准的 P/R/F1 指标。此时要求 llm_answer 和 std_answer 必须等长。
            适用于分类任务;

            当 ordered 为 False 时，计算nltk标准的 P/R/F1 指标。此时认为 llm_answer 和 std_answer 不计顺序，
            不必等长。若存在重复元素，则默认去重后计算。适用于关键词提取等任务

        ### smooth_threshould : float, default = 1.0, range = 0.0 ~ 1.0
            llm 的字符串输出往往存在偏差，按照exact match可能会出现score多为0的情况。例如
            candidate : ['大模型', '自然语言处理'], ref : ['大语言模型', '自然语言处理技术']。smooth_threshould
            给予了一定容忍阈值。设为1则不进行smooth，否则认为相似度大于阈值的一对字符串exact match.
        """
        llm_answer, std_answer = self.__pracf_check(
            llm_answer, std_answer, ordered, smooth_threshould)
        if ordered:
            return recall_score(std_answer, llm_answer, average=average)
        count = 0
        for i in range(len(std_answer)):
            if self.__gin(std_answer[i], llm_answer, smooth_threshould=smooth_threshould):
                count += 1
        return count / (len(std_answer) + 1e-8)

    def F1(
        self,
        llm_answer: List[Any],
        std_answer: List[Any],
        average: str = 'micro',
        ordered: bool = True,
        smooth_threshould=1.0,
        alpha=1
    ) -> float:
        """
        P/R/F1 计算
        ---------------
        ### ordered : bool : True | False, default = True
            当 ordered 为 True 时， 计算sklearn标准的 P/R/F1 指标。此时要求 llm_answer 和 std_answer 必须等长。
            适用于分类任务;

            当 ordered 为 False 时，计算nltk标准的 P/R/F1 指标。此时认为 llm_answer 和 std_answer 不计顺序，
            不必等长。若存在重复元素，则默认去重后计算。适用于关键词提取等任务

        ### smooth_threshould : float, default = 1.0, range = 0.0 ~ 1.0
            llm 的字符串输出往往存在偏差，按照exact match可能会出现score多为0的情况。例如
            candidate : ['大模型', '自然语言处理'], ref : ['大语言模型', '自然语言处理技术']。smooth_threshould
            给予了一定容忍阈值。设为1则不进行smooth，否则认为相似度大于阈值的一对字符串exact match.
        """
        p = self.Precision(llm_answer, std_answer, average,
                           ordered, smooth_threshould)
        r = self.Recall(llm_answer, std_answer, average,
                        ordered, smooth_threshould)
        f1 = (alpha**2 + 1) * p * r / (alpha**2 * p + r + 1e-8)
        return f1

    def Extraction_tri_eval(
        self,
        llm_answer: List[List[str]],
        std_answer: List[List[str]],
        return_dict: bool = False,
        detail: bool = False
    ):
        """
        事件抽取任务scorer
        ---------------
        输入：抽取结果(三元组)：List[str,str,str]，Ground Truth(三元组)：List[str,str,str]

        输出： F1-scores of trigger extraction, argument extraction and triplet extraction. 格式为

                词典{'trg_f' : trg_f, 'arg_f' : arg_f, 'trp_f' : trp_f}, return_dict = True

                平均值, return_dict = False

        Examples：
        >>> Extraction_tri_eval([['A','B','C'],['AA','BB','CC']],
                                [['A','B','C'],['AAA','BBB','CC']])


        """
        data = pd.DataFrame()
        data['pred_event_triples'] = [llm_answer]
        data['event_triples'] = [std_answer]
        trg_f, arg_f, trp_f = '0', '0', '0'
        try:
            trg_f, arg_f, trp_f = extract_tri_eval.evaluate(
                df=data, detail=detail)
        except:
            print(
                f"llm answer is : {llm_answer}, std answer is : {std_answer}")
            print("Failed to decode due to weird answer given by llm.")
        if return_dict:
            return {'trg_f': eval(trg_f), 'arg_f': eval(arg_f), 'trp_f': eval(trp_f)}
        else:
            return (eval(trg_f) + eval(arg_f) + eval(trp_f)) / 3
