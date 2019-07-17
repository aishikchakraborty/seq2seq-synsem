import numpy as np
data1 = np.random.randint(1, 10, size=(40040, 10)).tolist()
data2 = np.random.randint(1, 10, size=(600, 10)).tolist()
data1 = [[str(w) for w in d] for d in data1]
data2 = [[str(w) for w in d] for d in data2]

f = open('data/training/random.txt', 'w')
for i in range(40040):
    f.write('sos ' + ' '.join(data1[i]) + '\n')
f.close()
f = open('data/dev/random.txt', 'w')
for i in range(600):
    f.write('sos ' + ' '.join(data2[i]) + '\n')
f.close()
