import numpy as np
import matplotlib.pyplot as plt
import heapq
from numpy.linalg import pinv,inv,matrix_rank
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from libsvm.svmutil import *

c_list = []
gamma_list = []
for i in range(-5,16,2):
    c_list.append(2**i)
for i in range(-15,3,2):
    gamma_list.append(2**i)

def generate_data(num):
    x = np.linspace(-100,100,num)
    rd = np.random.normal(0,1,num)
    y = 2*x + rd
    x_list = []
    for i in range(num):
        x_list.append({1:x[i],2:y[i]})
    y_list = []
    for el in rd:
        if el < 0:
            y_list.append(-1)
        else:
            y_list.append(1)

    return x_list,y_list 
def five_fold(x,y):
    accuracy_without_scaling = []
    kf = KFold(5)
    for i,(train_idx,test_idx) in enumerate(kf.split(x)):
        train_x = [x[i] for i in range(len(x)) if i in train_idx]
        train_y = [y[i] for i in range(len(y)) if i in train_idx]

        test_x = [x[i] for i in range(len(x)) if i in test_idx]
        test_y = [y[i] for i in range(len(y)) if i in test_idx]
        for c in c_list:
            for g in gamma_list:
                params = f'-c {c} -g {g}'
                m = svm_train(train_y,train_x,params)
                _,p_acc,_ = svm_predict(test_y,test_x,m)
                accuracy_without_scaling.append([p_acc[0],c,g,i])
    
    accuracy_without_scaling.sort(reverse = True)

    accuracy = []
    train_x = [x[i] for i in range(len(x)) if i in train_idx]
    train_y = [y[i] for i in range(len(y)) if i in train_idx]

    test_x = [x[i] for i in range(len(x)) if i in test_idx]
    test_y = [y[i] for i in range(len(y)) if i in test_idx]
    
    for i in range(len(train_x)):
        for k,v in train_x[i].items():
            train_x[i][k] = v/10

    for i in range(len(test_x)):
        for k,v in test_x[i].items():
            test_x[i][k] = v/10

    params = f'-c {accuracy_without_scaling[0][1]} -g {accuracy_without_scaling[0][2]}'
    m = svm_train(train_y,train_x,params)
    _,p_acc,_ = svm_predict(test_y,test_x,m)
    accuracy.append([p_acc[0],accuracy_without_scaling[0][1],accuracy_without_scaling[0][2],i])

    print (f'Top 3 accuracy without scaling')
    for i in range(3):
        print(f'Top {i+1} accuracy : {accuracy_without_scaling[i][0]}%, c : {accuracy_without_scaling[i][1]}, gamma : {accuracy_without_scaling[i][2]}')
    
    print (f"Accuracy with scaling at Problem 3's parameters")
    print(f'accuracy : {accuracy[0][0]}%, c : {accuracy[0][1]}, gamma : {accuracy[0][2]}')