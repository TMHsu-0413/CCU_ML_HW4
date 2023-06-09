import numpy as np
import matplotlib.pyplot as plt
import random
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
    r_idx = []
    for i in range(50):
        r = random.randint(0,500)
        r_idx.append(r)
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    
    for i in range(500):
        if i in r_idx:
            train_x.append(x_list[i])        
            train_y.append(y_list[i])
            plt.plot(x[i],y[i],'o',color='red',markersize=2.0)
        else:
            test_x.append(x_list[i])
            test_y.append(y_list[i])
            plt.plot(x[i],y[i],'o',color='blue',markersize=2.0)

    plt.show()
    return train_x,train_y,test_x,test_y

def five_fold(x,y,tx,ty):
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

    for i in range(len(train_x)):
        for k,v in train_x[i].items():
            x[i][k] = v/10

    params = f'-c {accuracy_without_scaling[0][1]} -g {accuracy_without_scaling[0][2]}'
    m = svm_train(y,x,params)
    _,p_acc,_ = svm_predict(y,x,m)
    accuracy.append([p_acc[0],accuracy_without_scaling[0][1],accuracy_without_scaling[0][2],i])

    print (f'Top 3 accuracy without scaling')
    for i in range(3):
        print(f'Top {i+1} accuracy : {accuracy_without_scaling[i][0]}%, c : {accuracy_without_scaling[i][1]}, gamma : {accuracy_without_scaling[i][2]}')
    
    print (f"Accuracy with scaling at Problem 3's parameters")
    print(f'accuracy : {accuracy[0][0]}%, c : {accuracy[0][1]}, gamma : {accuracy[0][2]}')