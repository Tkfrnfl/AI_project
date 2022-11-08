from cmath import nan
from math import sqrt
import math
import numpy as np
from PIL import Image 
from random import random
import matplotlib.pyplot as plt

class NN:
    def __init__(self,input_data_posi,input_data_nega,country):

        input_data=[]
        out_data=[]
    
        learning_rate = 0.00001
        epochs = 500

        cost_list=[[0]*1000 for _ in range(10)]
        correct=0
        sum=0

        w1=np.random.rand(784,10)    # 가중치와 편향 초기화
        b1=np.random.rand(10)

        for i,val in enumerate(input_data_posi):  #각 나라별 input, output 데이터 정제
            if country in val[0]:
                tmp=val[0]
                input_data.append(tmp[2:])
                out_data.append(tmp[1:2])

        tmp_y=np.max(train_label)+1   #output one hot encoding
                y=np.eye(tmp_y)[train_label]
                x=np.array(train_list)

    def predict(self,x):
        W1,W2=self.params['W1'],self.params['W2']
        b1,b2=self.params['b1'],self.params['b2']
        
        a1=np.dot(x,W1) +b1
        z1=sigmoid(a1)
        a2=np.dot(z1,W2)+b2
        y=softmax(a2)
        
        return y         


