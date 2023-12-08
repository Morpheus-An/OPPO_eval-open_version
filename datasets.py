import os
import sys
from abc import abstractmethod


from commons.common_import import *
del EvalFuncDict

path_sep = '\\' if sys.platform == 'win32' else '/'

current_dir_path = os.path.dirname(__file__) + path_sep
data_dir_path = current_dir_path + "data" + path_sep

AvailableDatasets: Dict[str, 'BaseDataset'] = {}


class BaseDataset:
    _rel_path: str  # relative path of the current dataset
    _subset_dict: Dict['str', List[Tuple[Any, Any]]]  # available subsets
    _eval_func_dict: Dict[str, EvalFuncType]  # available evaluation functions

    Description: Dict[str, str] = {}

    TaskDescription: str = ''
    QuestionPrompt: str = ''

    def __new__(cls):
        ''' Singleton Pattern '''
        global EvalFuncDict
        obj = super().__new__(cls)
        if obj in AvailableDatasets.values():
            raise TypeError(
                f'Cannot create objects of singleton class {cls.__name__}!')
        if cls.__name__ in AvailableDatasets:
            raise NameError(
                f"Multiple dataset have the same name \"{cls.__name__}\"")
        AvailableDatasets[cls.__name__] = obj
        # origin_eval_func_dict = cls._eval_func_dict
        # cls._eval_func_dict = origin_eval_func_dict.copy()
        # for name in list(origin_eval_func_dict.keys()):
        #    del origin_eval_func_dict[name]
        # print(cls.__name__, cls._eval_func_dict)
        obj.__init__()

    def __init__(self):
        self.info: Optional[List[Any]] = None

    def load(self, subset_name, sample_num=None):
        '''load a subset; store it in self._subset_dict'''
        random.seed(1234)
        with open(self._rel_path + subset_name, "r", encoding='utf-8') as fp:
            self._subset_dict[subset_name] = self.analyse_file(
                fp, sample_num=sample_num)
        random.shuffle(self._subset_dict[subset_name])

    # NOTE: NOT used?
    def __getitem__(self, subset_name: str):
        '''read a subset rather than a sample'''
        return self._subset_dict[subset_name]

    def get_subset(self, subset_name: str):
        d = self._subset_dict[subset_name]
        if not d:
            self.load(subset_name)
            d = self._subset_dict[subset_name]
        return d

    @abstractmethod
    def analyse_file(self, fp: TextIO, sample_num: Optional[int] = None) -> List[Tuple[Any, Any]]:
        """
        :param fp: 带读取数据集的指针，以文本模式打开，utf-8编码
        :param sample_num: 希望从数据集中读取样例数目，None表示全部读取
        :return: 格式化后的样例列表。每个样例分为两部分，一部分用来生成prompt，一部分为标准答案，每部分具体格式不限制。
        """
        pass
# gruv box 
    @abstractmethod
    def fill_prompt(self, sample: Any, answer: Any) -> Dict[str, str]:
        """
        :param sample: 一个样例
        :param answer: 该样例对应的答案
        :return: 文件开头Prompt模板中有一些未填充的变量，这里需返回一个字典，以这些变量名称为键，以该样例的信息为值
        """
        pass

    @abstractmethod
    def validate(self, llm_answer: str, std_answer: Any) -> ScoreType:
        """
        :param llm_answer: 大模型给出的答案
        :param std_answer: 标准答案
        :return: 对模型给出的答案的评分，一般为布尔值或分数，也可以为其他
        注：该函数仅用于评判llm答案的准确性，其他指标请在 _eval_func_dict 中实现
        """
        self.info = ...  # placeholder
        # self.info 需要传回一些可能用于其他指标计算的信息，根据数据集题型不同而有差异
        # 后续处理中，该变量的多个属性将被分别组成列表。例，若传回(int, str)，则会变为(List[int], List[str])，然后交由 eval_func处理
        # 对于(多)分类问题，应当传回llm答案中的类别编号（无法提取的返回-1）和标准答案的类别编号。
        # 对于选择题，应当传回 True 或 False （是否选择正确）
        # 对于生成题或其它类型题目，根据需要传回特定信息
        pass

    def evaluate(self,
                 llm_func: Callable[[str], str],
                 llm_name: str,
                 subset_name: str, sample_num: int = None,
                 eval_funcs: Optional[Union[EvalFuncType,
                                            List[EvalFuncType], Dict[str, EvalFuncType]]] = None,
                 icl_sample_file: Optional[str] = None,
                 n_shot: Union[Literal[0], Literal[1], int] = 0) -> Any:
        """
        :param llm_func: 大模型的接口函数，要求输入一段prompt，能返回一段文字答案
        :param subset_name: 使用哪个子集（训练、测试、验证）测试大模型
        :param sample_num: 需要测试该子集中的多少个样本
        :param eval_funcs: 评估函数，计算除得分之外的一些其他指标。
        :param icl_sample_file: icl示例文件名称，该文件应当存储着作为示例的样例。
        :param n_shot: 需要提供的示例数目。
        :param save_detail: 是否输出细节，如果是的话，每个样例的测试过程都会输出。
        :return: 本次测试中大模型的最终得分
        """

        # 缺省参数的处理
        if eval_funcs is None:
            eval_funcs = self._eval_func_dict
        elif not isinstance(eval_funcs, list):
            eval_funcs = {func.__name__: func for func in eval_funcs}
        elif not isinstance(eval_funcs, dict):
            eval_funcs = eval_funcs
        else:
            eval_funcs = {eval_funcs.__name__: eval_funcs}

        # read test samples
        sample_list = self.get_subset(subset_name)[:sample_num]

        # reads ICL samples
        assert type(n_shot) == int and n_shot >= 0
        if n_shot > 0:
            if icl_sample_file is None:  # use default ICL file path
                for name in self._subset_dict:
                    if name[:12] == 'icl_samples.':  # 检查数据集文件夹中是否有名为 'icl-samples.xxx' 的文件
                        icl_sample_file = name
                        break
            assert icl_sample_file is not None, \
                "Requires {n_shot} ICL samples but NOT ICL sample file is found!"
            _icl_samples = self.get_subset(icl_sample_file)
            assert len(_icl_samples) >= n_shot, \
                f"Requires {n_shot} icl samples, but only {len(_icl_samples)} is available"
            # use the first `n_shot` samples
            _icl_samples = _icl_samples[:n_shot]
        else:
            _icl_samples = []

        # 初始化一些变量
        score_list = []
        info_dict: Dict[str, List[List[Any]]] = {}
        self.info = None
        detail = {
            "dataset": self.Description,
            "model": llm_name,
            "n_shot": n_shot,
            "conclusion": None,
            "record": [],
            "icl_samples": _icl_samples  # record the ICL samples
        }

        # try: ... except: ... finally: ... 语句保证了即使出现错误，也会被记录
        # Let's test it!
        try:
            for (sample, std_answer) in tqdm(sample_list):
                ## test begins ##
                # build prompt
                demon_prompt = ''
                if n_shot > 0:
                    kwargs_list = [self.fill_prompt(*s) for s in _icl_samples]
                    demon_prompt = "\n".join([
                        self.QuestionPrompt.format(**d) for d in kwargs_list
                    ])
                q_kwargs = self.fill_prompt(sample, std_answer)
                q_kwargs['answer'] = ''
                question_prompt = "\n" + self.QuestionPrompt.format(**q_kwargs)
                # full prompt = task description + icl samples + test sample
                prompt = self.TaskDescription + demon_prompt + question_prompt

                # 测试大模型，评分并记录
                # TODO: batch evaluation support
                llm_answer = llm_func(prompt).strip()
                score = self.validate(llm_answer, std_answer)
                if isinstance(score, bool) or isinstance(score, int):
                    score = float(score)
                score_list.append(score)

                # 检查validate函数中是否向info传入了数据，讲这些数据分条拼合，用于之后计算其它额外指标
                if self.info is not None:
                    for name in self.info:
                        if name not in info_dict:
                            # print("info:", self.info)
                            info_dict[name] = [[]
                                               for _ in range(len(self.info[name]))]
                    for name in eval_funcs:
                        if name in self.info:
                            for j, _sub_info in enumerate(self.info[name]):
                                info_dict[name][j].append(_sub_info)
                    self.info = None  # reset info

                # save the details
                # TODO: 记录 validate 函数解析的结果，帮助 debug
                #       比如模型回答 “A。”，解析结果为 “A”
                detail['record'].append({
                    'sample': sample,
                    'prompt_params': q_kwargs.copy(),
                    "prompt": prompt,
                    "llm_answer": llm_answer,
                    "std_answer": std_answer,
                    "score": score,
                })
                ## test ends ##
        except Exception as ex:
            detail['exception'] = f'{ex}'
            raise ex
        finally:
            "计算平均分和各个额外指标"
            avg_score = Functions.average(score_list, float('nan'))
            time = time_now()
            detail['time'] = time
            detail["conclusion"] = {
                "_avg_score": avg_score,
                "_count": len(score_list),
                **{name: eval_funcs[name](self, *info_dict[name]) for name in eval_funcs},
            }
            # print the conclusions
            print(detail['conclusion'])
            result_path = 'results' + self._rel_path[4:]
            if not os.path.exists(result_path):
                os.makedirs(result_path, mode=0o777)
            p = result_path + f'{llm_name}-{time}.json'
            dump_json(detail, p)
            print(p)

        return avg_score


BaseDataset.__name__ = 'Dataset'


def build_dataset_recursively(dir_path, cls):
    _list = os.listdir(dir_path)
    # NOTE: 过滤掉以 . 开头的隐藏文件
    _list = list(filter(lambda x: x[0] != '.', _list))
    _globals = globals() | {"ParentDataset": cls}
    del _globals['os']
    del _globals['abstractmethod']
    _locals: Dict[str, Any] = {'TaskDescription': '',
                               'QuestionPrompt': '', "EvalFuncDict": {}}
    if 'TODO' in _list:
        return
    if "config.py" in _list and os.path.isfile(dir_path + "config.py"):
        with open(dir_path + "config.py", "r", encoding='utf-8') as fp:
            code = (''.join(fp.readlines()))
    else:
        code = "TaskDescription = ParentDataset.TaskDescription\n"\
               "QuestionPrompt = ParentDataset.QuestionPrompt\n"\
               "class Dataset(ParentDataset):\n    pass\n"
    exec(code, _globals, _locals)
    _dataset: BaseDataset = _locals["Dataset"]
    _dataset.TaskDescription = _locals["TaskDescription"]
    _dataset.QuestionPrompt = _locals["QuestionPrompt"]
    _dataset._eval_func_dict = _locals["EvalFuncDict"].copy()
    _dataset._rel_path = os.path.relpath(dir_path, current_dir_path) + path_sep
    if 'name' in _dataset.Description:
        _dataset.__name__ = _dataset.Description['name']
    else:
        _dataset.__name__ = _dataset._rel_path.replace(path_sep, '_')[:-1]
    # print(_dataset._rel_path)
    _dataset._subset_dict = {}
    # treat all files other the .py files as the data file
    for name in _list:
        if os.path.isfile(dir_path + name) and name[-3:] != '.py':
            _dataset._subset_dict[name] = []
    if _dataset._subset_dict:
        # print(_dataset._subset_dict)
        _dataset()
    for name in _list:
        if os.path.isdir(dir_path + name + path_sep):
            build_dataset_recursively(dir_path + name + path_sep, _dataset)


build_dataset_recursively(data_dir_path, BaseDataset)


def ShowAvailableDatasets():
    print("Available Datasets:")
    for i, name in enumerate(AvailableDatasets):
        print(f'{i + 1}. [\033[1;32m{name}\033[0m]', end='')
        description = AvailableDatasets[name].Description
        if 'judge' in description:
            print(f'  -  judged by {description["description"]}', end='')
        if 'description' in description:
            print(f'  -  {description["description"]}', end='')
        print(end='\n')
