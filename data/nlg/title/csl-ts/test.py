def read_file(fp, sample_num=None):
    if sample_num == None:
            sample_num = 10e19
    dataset = []
    while line := fp.readline():
        if len(dataset) > sample_num:
            break
        _, content, title = line.strip().split('\t')
        dataset.append(((content, ), title))
    return dataset
with open(r'data\nlg\title\csl-ts\dev.tsv', 'r', encoding='utf-8') as fp:
    dataset = read_file(fp)
print(dataset[0])
