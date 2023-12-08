import seaborn as sns

with open(r'dev.tsv', 'r', encoding='utf-8') as f:
    lines = f.readlines()

n = [len(e) for e in lines]


sns.histplot(n)
