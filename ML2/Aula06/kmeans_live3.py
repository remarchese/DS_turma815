# -*- coding: utf-8 -*-
'''
K-Means com Iris
'''
#-----------------------------------------------------------------------------------------------
#bibliotecas
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_samples, silhouette_score
#-----------------------------------------------------------------------------------------------
iris = datasets.load_iris()
#-----------------------------------------------------------------------------------------------
#iris tem 4 features
numAmostras = len(iris.data)
features = np.zeros((numAmostras,2))
features[:,0] = iris.data[:,0]
features[:,1] = iris.data[:,2]

scaler = StandardScaler().fit(features)
features = scaler.transform(features)

#método do cotovelo
#consiste em analisar a distância quadrática dos pontos de um grupo para o seu centróide
kmeansModel = KMeans(n_clusters=1, random_state=0, max_iter=500)
kmeansModel.fit(features)
#inertia corresponde a soma quadrática das distâncias de cada ponto para o centróide
#do grupo ao qual pertence
print(kmeansModel.inertia_) 

#avaliar pelo método do cotovelo
inertia = [] #array para armazenar os valores de inertia
for k in range(1,15):
	kmeansModel = KMeans(n_clusters=k, random_state=0, max_iter=500)
	kmeansModel.fit(features)
	inertia.append(kmeansModel.inertia_)

#pelo gráfico gerado, imaginamos que um número k=3 ou k=4 parece
#ser interessante para o nosso problema
numClusters = 3
kmeansModel = KMeans(n_clusters=numClusters, random_state=0, max_iter=500)
kmeansModel.fit(features)


plt.figure()
plt.scatter(features[:,0],features[:,1])

plt.figure()
plt.plot(range(1,15),inertia)
plt.scatter(range(1,15),inertia,color='red',marker='*')
plt.xlabel('Número de clusters (k)')
plt.ylabel('Inertia')

colors = ['green','blue','red']
plt.figure()
for i in range(numAmostras):
	plt.scatter(features[i,0],features[i,1],color=colors[kmeansModel.labels_[i]])
plt.show()