# -*- coding: utf-8 -*-
'''
script para testar o método K-Means
'''
#bibliotecas
import numpy as np 
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from copy import copy

#1) parâmetros
numFeatures = 2 #define o espaço de features
numClusters = 2 #define o número de clusters
numSamples = 100 #define o número de amostras
maxIt = 300 #número máximo de iterações
itCounter = 0 #contador de iterações

#2) gerando as sementes
seeds = [] #array para guardar as coordenadas das sementes
for i in range(numClusters):
	seedv = np.zeros((numFeatures))
	for j in range(numFeatures):
		seedv[j] = np.random.normal(0.2,0.1) #ponto do centroide
	seeds.append(seedv) #guardar a semente
# print(seeds) #debugging 

#3) gerando o espaço de features
#Array para armazenar as features
features = np.zeros((numSamples,numFeatures)) 
#gerar as amostras
for i in range(numSamples):
	for j in range(numFeatures):
		if i < numSamples/2:
			features[i,j] = np.random.normal(0.2,0.1) #valor da feature
		else:
			features[i,j] = np.random.normal(0.8,0.05) #valor da feature

# print(features) #debugging

#4) clustering
#clusters são inicializados nas coordenadas das sementes
clustersCentroids = copy(seeds)
clustered = np.zeros((numSamples,1)) #array para armazenar os pontos que pertencem a um dado cluster
clusterList = [copy(clustersCentroids)] #lista para armazenar os centroides atuais
while True:
	#primeiro, atribuímos às amostras para algum cluster
	#baseado na distância euclidiana entre a amostra e os
	#clusters
	for i in range(numSamples):
		#array para armazenar a distância e o id do cluster
		arrayDistCluster = np.zeros((numClusters))
		for j in range(numClusters):
			#calcular a distância euclidiana entre a amostra
			#e o cluster
			arrayDistCluster[j] = euclidean(features[i,:],clustersCentroids[j])
			#print(i,j,arrayDistCluster) #debugging

		#ordenar as distâncias e identificar a qual cluster pertence a amostra
		clustered[i] = np.argsort(arrayDistCluster)[0]
	
	#agora que cada amostra foi atribuída para um cluster
	#vamos atualizar os centróides dos clusters
	old_clusterCentroids = copy(clustersCentroids)
	for k in range(numClusters): #vamos visitar os clusters
		#selecionar apenas as amostras que pertencem ao cluster
		sampBelongCluster = features[np.where(clustered == k)[0],:]
		# print(sampBelongCluster) #debugging
		if(len(sampBelongCluster) == 0):
			for j in range(numFeatures):
				clustersCentroids[k][j] = np.random.normal(0.2,0.1)
		else:
			#centróide é o ponto médio das features que fazem parte dele
			clustersCentroids[k] = np.mean(sampBelongCluster,axis=0)
			#print(seeds[k],clustersCentroids[k]) #debugging

	#insere os novos centroides calculados na lista
	clusterList.append(copy(clustersCentroids))
	
	itCounter += 1
	print(itCounter)
	if itCounter >= maxIt:
		break

#5) plotting
plt.figure()
for i in range(numSamples):
	if clustered[i] == 0: #pegando as amostras depois de agrupadas
		plt.scatter(features[i,0],features[i,1],color='blue',marker='*')
	else:
		plt.scatter(features[i,0],features[i,1],color='red',marker='*')

plt.scatter(seeds[0][0],seeds[0][1],color='black',marker='+')
plt.scatter(seeds[1][0],seeds[1][1],color='black',marker='+')

plt.scatter(clustersCentroids[0][0],clustersCentroids[0][1],color='blue',marker='+')
plt.scatter(clustersCentroids[1][0],clustersCentroids[1][1],color='red',marker='+')
plt.xlim([-0.05,1])
plt.ylim([-0.05,1])

#plt.scatter(features[:,0],features[:,1],color='black',marker='*')
#[plt.scatter(scenter[0],scenter[1]) for scenter in seeds]
#[plt.scatter(scenter[0],scenter[1]) for scenter in clustersCentroids]

plt.figure()
for i in range(len(clusterList)):
	plt.scatter(clusterList[i][0][0],clusterList[i][0][1],color='blue')
	plt.scatter(clusterList[i][1][0],clusterList[i][1][1],color='red')
plt.xlim([-0.05,1])
plt.ylim([-0.05,1])
plt.show()
