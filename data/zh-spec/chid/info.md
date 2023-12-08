chid 数据集的 [github](https://github.com/thu-coai/ChID-Dataset/blob/master/README.md) 和 [huggingface](https://huggingface.co/datasets/thu-coai/chid/tree/main/original)。

该数据集的test集分为四个；

- test.txt: IN-Domain 测试集，7个选项中有1个正确答案，3个相似成语，和3个随机抽取的成语；
- test_out.txt: Out-of-Domain 测试集，在篇幅、词频、文本来源等方面和训练集有差异；
- test_ran.txt: 备选项的设置和test集不同，采用完全随机的方式从成语词库中抽取6个；
- test_sim.txt: 备选项的设置和test集不同，从最相似的10个词中抽取6个。

