from func import *
from libsvm.svmutil import *

train_x,train_y,test_x,test_y = generate_data(500)
m = svm_train(train_y,train_x)
_,p_acc,_ = svm_predict(test_y,test_x,m)

five_fold(train_x,train_y,test_x,test_y)

print('accuracy without any parameter',str(p_acc[0]) + '%')