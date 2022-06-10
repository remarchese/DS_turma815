# -*- coding: utf-8 -*-

#Notes
'''
The research on K-means can be traced back to the middle of
the last century, conducted by numerous researchers across different disciplines,most
notably Lloyd (1957, 1982) [59], Forgey (1965) [35], Friedman and Rubin (1967)
[37], and MacQueen (1967) [63]. Jain and Dubes (1988) provides a detailed history
of K-means along with descriptions of several variations [48]. Gray and Neuhoff
(1998) put K-means in the larger context of hill-climbing algorithms [40].
'''

# Relembrando Aprendizado Não-Supervisionado
'''
 Aprendizado Supervisionado: Classificação / Regressão
 Premissa: Eu já conheço de antemão as classes

 Aprendizado Não Supervisionado: 
 Premissa: Eu não conheço de antemão as classes ou targets
 Os métodos de aprendizado não supervisionado vão nos ajudar a 
 descobrir padrões inerentes aos dados que não conhecemos de
 antemão
'''

'''
 K-Means vai nos ajudar a agrupar dados de acordo com similaridades
 Algoritmo:
  1) definir o número de grupos (clusters) que eu espero encontrar
  nos meus dados. gera pontos aleatórios que representam os centróides
  de um dado grupo.
  2) processo iterativo - objetivo: minimizar a distância entre vários
  pontos no espaço de características até o centróide do cluster
   2.1) primeira etapa: atribuição. nesta etapa, nós vamos
   atribuir os pontos no espaço de características para algum
   cluster
   2.2) para cada cluster, a gente atualiza o seu centróide para
   como sendo o ponto médio de todos os pontos que estão mais próximos
   a ele
   2.3) regra para encerrar o processo:
    - número máximo de iterações
    - estabelecer uma tolerância sendo que: caso a variação
    dos centroides seja pequena o suficiente, entendemos
    que houve uma convergência

'''

'''
Exemplo K-Means
'''
#bibliotecas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans #KMeans --> dentro das bibliotecas referentes clusterização
from sklearn.metrics import silhouette_samples, silhouette_score

#parâmetros
numAmostras = 1000
numClusters = 3
numFeatures = 2

#1) gerando pontos aleatórios
matFeatures = np.zeros((numAmostras, numFeatures)) #matriz bidimensional com N amostras
for i in range(numAmostras): #varrer o total de amostras
	for j in range(numFeatures): #varrer o total de características
		if i < numAmostras/2:
			#pontos aleatórios dentro de uma distribuição normal
			matFeatures[i,j] = np.random.normal(0.1,0.3)
		else:
			#pontos aleatórios dentro de uma distribuição normal
			matFeatures[i,j] = np.random.normal(0.2,0.1)


#2) aplicar o kmeans
#criar o modelo
kmeansModel = KMeans(n_clusters=numClusters, random_state=0, max_iter=1000)
#etapa de encontrar os grupos e os centróides que representam os clusters
kmeansModel.fit(matFeatures) 

print(kmeansModel.labels_) #printar o Id das classes
print(kmeansModel.cluster_centers_) #printar os centroides dos clusters

plt.figure()
plt.scatter(matFeatures[:,0], matFeatures[:,1], color='black', marker='*')			

plt.figure()
for i in range(numAmostras):
	if kmeansModel.labels_[i] == 0:
		plt.scatter(matFeatures[i,0],matFeatures[i,1],color='blue',marker='*')
	elif kmeansModel.labels_[i] == 1:
		plt.scatter(matFeatures[i,0],matFeatures[i,1],color='green',marker='*')
	else:
		plt.scatter(matFeatures[i,0],matFeatures[i,1],color='red',marker='*')


plt.scatter(kmeansModel.cluster_centers_[0][0],kmeansModel.cluster_centers_[0][1],color='blue',marker='+')
plt.scatter(kmeansModel.cluster_centers_[1][0],kmeansModel.cluster_centers_[1][1],color='green',marker='+')
plt.scatter(kmeansModel.cluster_centers_[2][0],kmeansModel.cluster_centers_[2][1],color='red',marker='+')
plt.show()