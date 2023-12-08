import random
random.seed(1234)

with open(r'data\nlg\title\csl-ts\dev.tsv', 'r', encoding='utf-8') as f:
    lines = f.readlines()
samples = random.sample(lines, k=50)

with open(r'data\nlg\title\csl-ts\icl_samples.tsv', 'w', encoding='utf-8') as f:
    f.writelines(samples)
