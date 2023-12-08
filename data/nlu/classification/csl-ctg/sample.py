import random
random.seed(1234)

with open('./train.tsv', 'r') as f:
    lines = f.readlines()
samples = random.sample(lines, k=50)

with open('./icl_samples.tsv', 'w') as f:
    f.writelines(samples)
