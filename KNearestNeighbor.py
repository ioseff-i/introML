import pandas as pd
from matplotlib import pyplot as plt
import math
import random
import time
random.seed(time.time())
def euclidean_distance(ls1,ls2):
    distance = 0.0
    for i in range(len(ls1)):
        distance += (ls1[i]-ls2[i])**2
    return math.sqrt(distance)

def get_neighbors(train,test_raw,num_neighbors):
    distances = list()
    for train_raw in train:
        dist = euclidean_distance(test_raw,train_raw)
        distances.append((train_raw,dist))
    distances.sort(key = lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors
def prediction(train,test_raw,num_neighbors):
    neighbors = get_neighbors(train,test_raw,num_neighbors)
    outputs = [row[-1] for row in neighbors]
    prediction = max(set(outputs),key = outputs.count)
    return prediction
def def_correctness(test,k):
    num_corr = 0
    for i in range(len(test)):
        predict = prediction(test,test[i],k)
        if(predict==test[i][-1]):
            num_corr+=1
    return (num_corr/len(test))*100
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0
    
df = pd.read_csv('~/Downloads/MOCK_DATA.csv')
#print(df.head())
dataset=df.values.tolist()
print(dataset)

list_k = list(range(1,20))
list_pr = list()
for i in list_k:
    list_pr.append(def_correctness(dataset,i))
plt.plot(list_k,list_pr)
plt.show()


