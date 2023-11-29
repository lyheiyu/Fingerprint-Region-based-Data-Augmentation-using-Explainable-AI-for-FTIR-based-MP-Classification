import numpy
import numpy as np
from utils import utils
onelist=[]
twolist=[]
threelist=[]
onelist2=[]

def clusterIdx(clusterNumers,wavelength):
    firstidx=[]
    lastidx=[]
    cluserlength=0
    for i in range(len(clusterNumers)):
        spes=[]
        for j in range(len(clusterNumers[i])-1):
            cluserlength=len(clusterNumers[i])
            if clusterNumers[i][j]!=clusterNumers[i][j+1]:
                spes.append(j)
        firstidx.append(spes[0])
        lastidx.append(spes[-1])
    firstidx = np.array(firstidx)
    lastidx = np.array(lastidx)
    fposition=(firstidx.mean())/(cluserlength)
    lposition=(lastidx.mean())/(cluserlength)
    first=int(wavelength*fposition)
    second=int(wavelength*lposition)
    return first,second

for j in range(20):
    onelist.append(0)
    onelist2.append(0)
for j in range(20):
    twolist.append(1)
for j in range(20):
    threelist.append(2)
onelist=numpy.array(onelist)
onelist2=numpy.array(onelist2)
twolist=numpy.array(twolist)
threelist=numpy.array(threelist)
listtotal=np.concatenate((onelist,twolist),axis=0)
listtotal=np.concatenate((listtotal,onelist2),axis=0)
listtotal=np.concatenate((listtotal,threelist),axis=0)

listclusters=[]
for i in range(10):
    listclusters.append(listtotal)

fidx,lidx=clusterIdx(listclusters,1761)
fidx=np.array(fidx)
lidx=np.array(lidx)
print(fidx,lidx)
# polymerName, waveLength, intensity, polymerID, x_each, y_each = utils.parseData11('data/D4_4_publication11.csv', 2,1764)
# print(waveLength)

# for i in range(len(listtotal)):
#     idx1 = [idx1 for idx1, id in enumerate(listtotal) if id == 0]
#     # print(idx1)
#     sperate1 = []
#     # sperate1.append(idx1[0])
#     for j in range(len(idx1)-1):
#
#
#         if idx1[j]!=idx1[j+1]-1:
#             sperate1.append(idx1[j])
#             sperate1.append(idx1[j+1])
    # sperate1.append(idx1[-1])
# sperate = []
# for i in range(len(listtotal)-1):
#     idx1 = [idx1 for idx1, id in enumerate(listtotal)]
#     # print(idx1)
#
#     print(listtotal[i],listtotal[i+1])
#     if listtotal[i]!=listtotal[i+1]:
#         print(listtotal[i+1])
#         sperate.append(idx1[i])
# # print(idx1)
# # print(sperate1)
# print(sperate)    #
#