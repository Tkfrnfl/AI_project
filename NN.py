from cmath import nan
from math import sqrt
import math
import numpy as np
from PIL import Image 
from random import random
import matplotlib.pyplot as plt

class NN:
    def __init__(self,input_data_posi,input_data_nega,country):

        input_data_po=[]
        out_data_po=[]
        input_data_ne=[]
        out_data_ne=[]
    
        learning_rate = 0.00001
        epochs = 500

        cost_list=[[0]*1000 for _ in range(10)]
        correct=0
        sum=0

        w1=np.random.rand(784,10)    # 가중치와 편향 초기화
        b1=np.random.rand(10)

        for i,val in enumerate(input_data_posi):  #각 나라별 input, output 데이터 정제(posi)
            if country in val[0]:
                tmp=val[0]
                input_data_po.append(tmp[2:])
                out_data_po.append(tmp[1])

        out_data_po=list(map(int,out_data_po))

        tmp_ohe=np.max(out_data_po)+1          #output one hot encoding
        out_ohe_po=np.eye(tmp_ohe)[out_data_po]

        for i,val in enumerate(input_data_nega):  #각 나라별 input, output 데이터 정제(nega)
            if country in val[0]:
                tmp=val[0]
                input_data_ne.append(tmp[2:])
                out_data_ne.append(tmp[1])

        out_data_ne=list(map(int,out_data_ne))

        tmp_ohe=np.max(out_data_ne)+1          #output one hot encoding
        out_ohe_ne=np.eye(tmp_ohe)[out_data_ne]

    def predict(self,x):
        W1,W2=self.params['W1'],self.params['W2']
        b1,b2=self.params['b1'],self.params['b2']
        
        a1=np.dot(x,W1) +b1
        z1=sigmoid(a1)
        a2=np.dot(z1,W2)+b2
        y=softmax(a2)
        
        return y         


