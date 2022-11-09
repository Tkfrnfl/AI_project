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
        self.params={}
        self.params['w1']=np.random.rand(4,10)  
        self.params['b1']=np.random.rand(10)
        self.params['w2']=np.random.rand(10,10)  
        self.params['b2']=np.random.rand(10)
        # w1=np.random.rand(10,10)    # 가중치와 편향 초기화
        # b1=np.random.rand(10)
        # w2=np.random.rand(10,10)   
        # b2=np.random.rand(10)

        for i,val in enumerate(input_data_posi):  #각 나라별 input, output 데이터 정제(posi)
            if country in val[0]:
                tmp=val[0]
                input_data_po.append(tmp[2:])
                out_data_po.append(tmp[1])
             
        #input_data_po=list(map(int,input_data_po))  #int 형변환
        input_data_po=np.array(input_data_po,np.int32)
        print(input_data_po)
        out_data_po=list(map(int,out_data_po))  

        #tmp_ohe=np.max(out_data_po)+1          #output one hot encoding
        out_ohe_po=np.eye(10)[out_data_po]

        for i,val in enumerate(input_data_nega):  #각 나라별 input, output 데이터 정제(nega)
            if country in val[0]:
                tmp=val[0]
                input_data_ne.append(tmp[2:])
                out_data_ne.append(tmp[1])

        #input_data_ne=list(map(int,input_data_ne))  #int 형변환
        out_data_ne=list(map(int,out_data_ne))

        #tmp_ohe=np.max(out_data_ne)+1          #output one hot encoding
        out_ohe_ne=np.eye(10)[out_data_ne]
        
        #NN.predict(self,input_data_po[0])
        NN.numerical_gradient(self,input_data_po[0],out_ohe_po[0])
    

    def predict(self,x):
        w1,w2=self.params['w1'],self.params['w2']
        b1,b2=self.params['b1'],self.params['b2']
        a1=np.dot(x,w1) +b1
        z1=NN.sigmoid(self,a1)
        a2=np.dot(z1,w2)+b2
        y=NN.softmax(a2)

        return y

        
    
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def softmax(x):
        c=np.max(x,axis=0)
        exp_x=np.exp(x-c)
        y=exp_x/np.sum(exp_x,axis=0)
        return y
    
    def loss(self,x,t):
        y=self.predict(self,x)

        return self.cee(y,t)

    def cee(y,t):
        delta = 1e-7                  
        return -np.sum(t*np.log(y+delta))

    def numerical_gradient(f,x):
        h=1e-4
        grad=np.zeros_like(x)

        for idx in range(x.size):
            tmp_val=x[idx]

            x[idx]=tmp_val+h
            fxh1=f(x)

            x[idx]=tmp_val-h
            fxh2=f(x)

            grad[idx]=(fxh1-fxh2)/(2*h)
            x[idx]=tmp_val
        return grad    