from func import *
import numpy as np
import matplotlib.pyplot as plt
from libsvm.svmutil import *

x,y = generate_data(500)
m = svm_train(y,x)
