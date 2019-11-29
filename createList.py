import numpy as np

k = 5000
i = 0
tmp = []
tmp2 = []
while(i < k):
    tmp.append(0)
    tmp2.append(-3)
    i += 1
f = open('actionList.txt','w')
for i in tmp:
    f.write(str(i)+ "\n")
f.close()
f = open('rewards.txt','w')
for i in tmp2:
    f.write(str(i)+ "\n")
f.close()

