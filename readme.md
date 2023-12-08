# 公开数据评测框架

## 项目概况

该框架是为了方便数据集的评测设计, 提供了统一的接口和运行方式, 便于添加新数据集和评测.
项目目前适配了零样本和少样本(ICL)预测, 并添加了收集的公开数据集, 能支持基本的评测需求.
受限于时间, 尚有一些工作未完成:

1. 未较好地处理模型语境窗口长度 (Context length) 带来的影响, 部分数据集在进行 few-shot 评测时可能会出现输入过长的情况;
2. 尚未适配 Chain-of-Thought;
3. 尚未适配 LLM-as-a-judge 评测方法, 请参考 LLM-judge 项目;
4. 代码由多个成员共同完成, 在提示词设计, 代码风格, 框架设计等方面有差异, 也可能存在一些bug.

## 项目结构

项目主目录下有:

- `data/`: 数据集文件夹, 根据任务类型树状分类组织
- `results/`: 评测结果和中间数据目录, 结构组合和 data/ 一致
- `utils/`: 存放评测指标, 大模型接口等代码
- `commons/`: 常用工具函数
- `guides/`: 帮助理解项目和文档的示例代码等等材料
- `datasets.py`: BaseDataset 类代码文件
- `defaults.py`: 默认值配置文件
- `main.py`: 主文件
- `run.sh `和` run_all.sh`: 运行脚本, 内有注释

其中, data 文件夹根据任务和数据类型进行了组织:

- `data`:
    - `math`: 数学, 包括 `math23k` 和 `math401` 两个数据集
    - `nlu`: 自然语言理解
        - `classification`: 分类, 包括
            - `csl-tsg`
            - `csl-dcp`
        - `extraction`: 抽取, 包括
            - `csl-kg`: 关键词抽取
            - `cmrc`: 抽取式问答
            - `title2event`: 事件抽取
        - `match`: 匹配, 包括
            - `afqmc`: 语义匹配
        - `mrc`: 阅读理解, 包括
            - `c3-d`
            - `c3-m`
            - `ncr-xdw`
        - `ner`: 实体识别, 包括
            - `msra-ner`
    - `nlg`: 自然语言生成
        - `summarization`: 摘要, 包括
            - `vcsum-short`
        - `title`: 标题生成, 包括
            - `csl-t`
        - `translation`: 翻译, 包括
            - `wmt19-en-zh`
            - `wmt19-zh-en`
    - `rsn`: 逻辑推理, 包括
        - `LogiQA`
    - `klg`: 知识基准, 包括
        - `M3KE`: 学科数据
        - `ccs-commonsense`: 中文常识数据
        - `clue-wsc`: 常识推理数据
    - `zh-spec`: 中文特性
        - `ccs-sentence`: 行测语句表达
        - `ccs-word`: 行测词语表达
        - `chid`: 成语
        - `drcd`: 繁体字阅读
        - `ncr-gs`: 故事阅读
        - `ncr-wyw`: 文言文阅读

## 依赖

我们在 python3.9 和 python3.10 中进行了测试, 使用的第三方库有:

- nltk
- sklearn
- pandas
- numpy
- tqdm
- jieba
- tqdm

## 运行代码

代码使用 `openai` 的 API 接口进行了测试.
为正常使用 `openai` 的接口, 你需要添加 `openai` 的 key.
你可以在 `ChatGPT_API_Key.txt` 中粘贴 key,
也可以在 `utils/LLM/ChatGPT.py` 文件的开头添加.
请确保网络连接顺畅, 然后运行 `run.sh` 脚本测试:

```shell
./run.sh
```

如果缺少权限, 请运行:

```shell
chmod +x ./run.sh
```

成功运行的话, console 会打印评测参数, 评测进度条和评测结果. 
如需批量测试数据集, 运行 run_all.sh
```shell
./run_all.sh
```

## 缺失数据

目前还缺失以下数据集:

- `data/klg/clue-wsc/test.json`: 需要添加标签
- `data/klg/ccs-commonsense`: 行测中文常识
- `data/nlu/mrc/ccs-paragraph`: 行测篇章理解
- `data/zh-spec/ccs-word`: 行测词语运用
- `data/zh-spec/ccs-sentence`: 行测语句表达
- `data/numeric/ccs-stat-numeric`: 行测文字资料分析

这些数据集我们都在 data 目录下建立对应的文件夹占位.

## 数据集添加

在 `data/` 目录下,
每一个 **除了子文件夹和以 . 开头的隐藏文件外,
在该目录下包含其他文件** 的文件夹都被视作一个数据集.
其中, 包含一个 `TODO` 文件的文件夹及其子文件夹将被跳过,
这方便用户在 data/ 中添加其他功能的文件夹,
比如给缺失的数据集占位 (placeholder).

每个数据集文件夹下, 都要包含一个 `config.py` 文件.
这个文件是该数据集的配置文件,
确定该数据集的名称, 使用的提示词, 评分方法等各种有关信息.

代码在初始化时, 会递归地遍历 `data` 目录 (参见 `datasets.build_dataset_recursively` 函数),
跳过所有包含 `TODO` 文件的目录,
并读取其他目录下的 `config.py` 文件,
将数据集的配置读入内存.

因此, 往项目中添加新数据集的步骤是:

1. 在 `data/` 文件夹中的合适位置添加一个新的目录, 新建一个 `TODO` 文件
1. 往该文件夹中添加数据, 以及其他有关的文件
1. 在该文件夹中新建 `config.py` 文件, 编写有关配置
1. 在 `defaults.py` 文件中添加该数据集的默认评测集,
    为方便查阅, 也可以在 `run.sh` 和 `run_all.sh` 两个脚本中添加数据集名称
1. 删除文件夹中的 `TODO` 文件
1. 通过运行 `dataset.ShowAvailableDatasets` 函数查看数据集是否添加成功

`config.py`文件的编写我们在 `guides/config.py` 中提供了一个[模板](./guides/config.py),
`guides/config_simple.py` 则提供了一个[简化模板](guides/config_simple.py).
目前我们推荐参考简化模板编写.

## 大模型接口的添加

大模型接口统一存放于 `utils/LLM/` 文件夹中,
可以通过新建文件的形式添加新的大模型接口.
我们对大模型接口的具体实现没有要求,
只要求其接收一个字符串, 返回一个字符串, 相当于 `Callable[str, str]`.
之后, 你可以在 `defaults.py` 文件中添加默认评测接口,
或者修改 `main.py` 文件, 自己设计传参方式.

## 输入长度影响

1. csl-ts, csl-dcp 数据集的输入长度最长在 1200 字左右, 我们根据频率分布截断为600字;
2. cmrc 的输入长度最长接近 1000 字;
1. ncr-xdw 的输入长度最长超过 4000 字, 我们根据频率分布统一截断为 2000 字;
1. chid 数据集的输入长度最长接近 600 字;
1. drcd 数据集的输入长度最长接近 1000 字;
1. ncr-wyw 数据集的输入长度最长超过 1200 字;

考虑到 token 数量可能比实际字数更大,
使用语境窗口较小的模型进行测试时,
可能会出现长度超限的情况.

其他数据集即使是使用 1024 窗口长度的模型, 也应该能支持 3-5 shot 的预测.
