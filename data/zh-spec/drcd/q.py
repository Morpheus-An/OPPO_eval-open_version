import random

with open('dev.txt', 'r', encoding='utf-8') as fp:
    a = fp.readlines()
    w = []
    r = 80 / (len(a) // 1)
    for i in range(len(a) // 1):
        if random.random() < r:
            w.append(''.join(a[1*i: 1*(i+1)]))
    with open('icl_samples.txt', 'w', encoding='utf-8') as fp:
        for i in range(50):
            fp.write(w[i])
    print(len(w))