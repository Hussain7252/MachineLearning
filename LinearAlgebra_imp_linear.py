import numpy as np
"""
dic=[(0.10, 0.65),(0.50, 0.10),(0.90, 0.35),(-0.20, 0.17),(-0.5, 0.42),(1.50, 2.62)]
#First entry is the input variable X and second is the output variable Y
xlst=[]
ylst=[]
for  i in dic:
    x,y=i
    xlst.append(x)
    ylst.append(y)
xarr=np.array(xlst)
yarr=np.array(ylst)
MatrixA=np.matrix()
"""
A=np.matrix([[1, 0.10],[1,0.50],[1,0.90],[1,-0.20],[1,-0.5],[1,1.50]])
B=np.matrix([[0.65],[0.10],[0.35],[0.17],[0.42],[2.62]])
Atrans=A.transpose()
X=np.matmul(Atrans,A)
Y=np.matmul(Atrans,B)
print(X)
print(Y)