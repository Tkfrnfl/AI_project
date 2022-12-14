from cmath import nan
from math import sqrt
import math
import numpy as np
from PIL import Image 
from random import random
import matplotlib.pyplot as plt

class NN:
    plt_X=[]
    def __init__(self,input_data_posi,input_data_nega,country):

        input_data_po=[]
        out_data_po=[]
        input_data_ne=[]
        out_data_ne=[]


        lr = 0.05
        epochs = 10

        self.params={}
        self.params['w1']=np.random.rand(3,12)/np.sqrt(12)      # 가중치와 편향 초기화
        self.params['b1']=np.zeros(12)
        self.params['w2']=np.random.rand(12,12) /np.sqrt(12)  
        self.params['b2']=np.zeros(12)


        for i,val in enumerate(input_data_posi):  #각 나라별 input, output 데이터 정제(posi)
            if country in val[0]:
                tmp=val[0]
                input_data_po.append(tmp[2:])
                out_data_po.append(tmp[1])   
        #float 형변환
        input_data_po=np.array(input_data_po,np.float32)
        input_data_po=self.nomalize(input_data_po)
        out_data_po=list(map(int,out_data_po))  

        #output one hot encoding
        out_ohe_po=np.eye(12)[out_data_po]


        for i,val in enumerate(input_data_nega):  #각 나라별 input, output 데이터 정제(nega)
            if country in val[0]:
                tmp=val[0]
                input_data_ne.append(tmp[2:])
                out_data_ne.append(tmp[1])

        #float 형변환
        input_data_ne=np.array(input_data_ne,np.float32)
        #input_data_ne=self.nomalize(input_data_ne)
        out_data_ne=list(map(int,out_data_ne))

        #output one hot encoding
        out_ohe_ne=np.eye(12)[out_data_ne]
        
        #NN.predict(self,input_data_po[0])
        ans=0
        total=0
        for i in range(epochs):

            for i,val in enumerate(input_data_po):

                grad=NN.numerical_gradient(self,input_data_po[i],out_ohe_po[i])
                self.params['w1']-=lr*grad['w1']        #grad 값과 lr에 따라 가중치,편차값 조정 
                self.params['b1']-=lr*grad['b1']
                self.params['w2']-=lr*grad['w2']
                self.params['b2']-=lr*grad['b2']

            for i,val in enumerate(input_data_po):    
                ans+=self.accuracy(input_data_po[i],out_ohe_po[i])
                total+=1

        print('정확도:'+str(ans/total))     
        plt.plot(NN.plt_X)
        plt.show()

            

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
        y=NN.predict(self,x)

        return NN.cee(y,t)

    def cee(y,t):
        if y.ndim == 1:
                t = t.reshape(1, t.size)
                y = y.reshape(1, y.size)

            # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
        if t.size == y.size:
                t = t.argmax(axis=1)

        batch_size = y.shape[0]
        print('loss: '+str(-np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size))
        NN.plt_X.append(-np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size)
        return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
        



    def numerical_gradient_no_batch(f,x):
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

    def gradient(f, X):                 #x 배열 차원에 따라 다르게 gradient
        if X.ndim == 1:
            return NN.numerical_gradient_no_batch(f, X)
        else:
            grad = np.zeros_like(X)
            
            for idx, x in enumerate(X):
                grad[idx] = NN.numerical_gradient_no_batch(f, x)
            
            return grad


    def numerical_gradient(self,x,t):
        loss_W = lambda W: self.loss(x, t)   #람다 정규식> lambda 변수: return 식
        grads = {}
        grads['w1'] = NN.gradient(loss_W, self.params['w1'])
        grads['b1'] = NN.gradient(loss_W, self.params['b1'])
        grads['w2'] = NN.gradient(loss_W, self.params['w2'])
        grads['b2'] = NN.gradient(loss_W, self.params['b2'])
        return grads

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y) 
        t = np.argmax(t)

        if y==t :
            return 1
        else:
            return 0    

    def nomalize(self,x):
        min=np.min(x,axis=0)
        max=np.max(x,axis=0)
        
        for i,val in enumerate(x):
            x[i]=(x[i]-min)/(max-min)

        return x        