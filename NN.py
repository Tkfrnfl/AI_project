from cmath import nan
from math import sqrt
import math
import numpy as np
from PIL import Image 
from random import random
import matplotlib.pyplot as plt
from collections import OrderedDict

class NN:
    params={}
    plt_X=[]
    def __init__(self,input_data_posi,input_data_nega,country):

        network = TwoLayerNet(input_size=3, hidden_size=10, output_size=12)
        input_data_po=[]
        out_data_po=[]
        input_data_ne=[]
        out_data_ne=[]

        m = {}          #adam 에서 쓰일 m,v값
        m_hat = {}
        v = {}
        v_hat = {}

        m_b = {}        #adam 에서 쓰일 m,v값 for bias
        m_hat_b = {}
        v_b = {}
        v_hat_b = {}

        m2 = {}         #adam 에서 쓰일 m,v값 for w2
        m2_hat = {}
        v2 = {}
        v2_hat = {}

        m2_b = {}       #adam 에서 쓰일 m,v값 for w2 bias
        m2_hat_b = {}
        v2_b = {}
        v2_hat_b = {}

        beta_1=0.9
        beta_2=0.999
        epsilon=10e-6

        m=np.zeros( NN.params['W1'].shape)
        m_hat=np.zeros( NN.params['W1'].shape)
        v=np.zeros( NN.params['W1'].shape)
        v_hat=np.zeros( NN.params['W1'].shape)

        m_b=np.zeros( NN.params['b1'].shape)
        m_hat_b=np.zeros( NN.params['b1'].shape)
        v_b=np.zeros( NN.params['b1'].shape)
        v_hat_b=np.zeros( NN.params['b1'].shape)

        m2=np.zeros( NN.params['W2'].shape)
        m2_hat=np.zeros( NN.params['W2'].shape)
        v2=np.zeros( NN.params['W2'].shape)
        v2_hat=np.zeros( NN.params['W2'].shape)

        m2_b=np.zeros( NN.params['b2'].shape)
        m2_hat_b=np.zeros( NN.params['b2'].shape)
        v2_b=np.zeros( NN.params['b2'].shape)
        v2_hat_b=np.zeros( NN.params['b2'].shape)



        lr = 0.01
        epochs = 200


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
        
        ans=0
        total=0
        for i in range(epochs):

            for i,val in enumerate(input_data_po):
                
                grads=network.gradient(input_data_po[i],out_ohe_po[i])
                # NN.params['W1']-=lr*grads['W1']        #grad 값과 lr에 따라 가중치,편차값 조정 
                # NN.params['b1']-=lr*grads['b1']           # SGD기법
                # NN.params['W2']-=lr*grads['W2']
                # NN.params['b2']-=lr*grads['b2']

                
                ########adam#####
                for i,val in  enumerate(NN.params['W1']):   #Adam기법
                    for j,val_j in enumerate(val):
                        m[i][j]=beta_1*m[i][j]+(1-beta_1)*grads['W1'][i][j]
                        m_hat[i][j]=m[i][j]/(1-beta_1*beta_1)
                        v[i][j] = beta_2 * v[i][j] + (1 - beta_2) * grads['W1'][i][j] * grads['W1'][i][j]
                        v_hat[i][j] = v[i][j] / (1 - beta_2 * beta_2)
                        NN.params['W1'][i][j] -= lr * m_hat[i][j] / np.sqrt(v_hat[i][j] + epsilon)
                for i,val in enumerate(NN.params['b1']):
                        m_b[i]=beta_1*m_b[i]+(1-beta_1)*grads['b1']
                        m_hat_b[i]=m_b[i]/(1-beta_1*beta_1)
                        v_b[i] = beta_2 * v_b[i] + (1 - beta_2) * grads['b1'] * grads['b1']
                        v_hat_b[i] = v_b[i] / (1 - beta_2 * beta_2)
                        NN.params['b1'][i]-= lr * m_hat_b[i] / np.sqrt(v_hat_b[i] + epsilon)
                
                for i,val in  enumerate(NN.params['W2']):
                    for j,val_j in enumerate(val):
                        m2[i][j]=beta_1*m2[i][j]+(1-beta_1)*grads['W2'][i][j]
                        m2_hat[i][j]=m2[i][j]/(1-beta_1*beta_1)
                        v2[i][j] = beta_2 * v2[i][j] + (1 - beta_2) * grads['W2'][i][j] * grads['W2'][i][j]
                        v2_hat[i][j] = v2[i][j] / (1 - beta_2 * beta_2)
                        NN.params['W2'][i][j] -= lr * m2_hat[i][j] / np.sqrt(v2_hat[i][j] + epsilon)
                for i,val in enumerate(NN.params['b2']):
                        m2_b[i]=beta_1*m2_b[i]+(1-beta_1)*grads['b2']
                        m2_hat_b[i]=m2_b[i]/(1-beta_1*beta_1)
                        v2_b[i] = beta_2 * v2_b[i] + (1 - beta_2) * grads['b2'] * grads['b2']
                        v2_hat_b[i] = v2_b[i] / (1 - beta_2 * beta_2)
                        NN.params['b2'][i] -= lr * m2_hat_b[i] / np.sqrt(v2_hat_b[i] + epsilon)
                 ########adam#####

        for i,val in enumerate(input_data_po):
            tmp=network.accuracy(input_data_po[i],out_ohe_po[i])
            ans+=tmp
            total+=1
        plt.plot(NN.plt_X)
        plt.show()

        checkForWorldCup=np.array([45,3,6])     #점유율,유효슈팅,슈팅 ex)68,3,5
        checkForWorldCup=self.nomalize(checkForWorldCup)
        network.accuracy_woldcup(checkForWorldCup)

        print('정확도: '+str(ans/total))        

            
################################수치 미분법 코드###################
    # def predict(self,x):
    #     w1,w2=self.params['W1'],self.params['W2']
    #     b1,b2=self.params['b1'],self.params['b2']
    #     a1=np.dot(x,w1) +b1
    #     z1=NN.sigmoid(self,a1)
    #     a2=np.dot(z1,w2)+b2
    #     y=NN.sofracytmax(a2)

    #     return y

        
    
    # def sigmoid(self,x):
    #     return 1/(1+np.exp(-x))

    # def softmax(x):
    #     c=np.max(x,axis=0)
    #     exp_x=np.exp(x-c)
    #     y=exp_x/np.sum(exp_x,axis=0)
    #     return y
    
    # def loss(self,x,t):
    #     y=NN.predict(self,x)

    #     return NN.cee(y,t)

    # def cee(y,t):
    #     delta = 1e-7                  
    #     if y.ndim==1:
    #         t=t.reshape(1,t.size)
    #         y=y.reshape(1,y.size)
    #     print(-np.sum(t*np.log(y+delta))    )   
    #     return -np.sum(t*np.log(y+delta))    

    # def numerical_gradient_no_batch(f,x):
    #     h=1e-4
    #     grad=np.zeros_like(x)
    #     for idx in range(x.size):
    #         tmp_val=x[idx]

    #         x[idx]=tmp_val+h
    #         fxh1=f(x)
    #         x[idx]=tmp_val-h
    #         fxh2=f(x)

    #         grad[idx]=(fxh1-fxh2)/(2*h)
    #         x[idx]=tmp_val
    #     return grad    

    # def gradient(f, X):                 #x 배열 차원에 따라 다르게 gradient
    #     if X.ndim == 1:
    #         return NN.numerical_gradient_no_batch(f, X)
    #     else:
    #         grad = np.zeros_like(X)
            
    #         for idx, x in enumerate(X):
    #             grad[idx] = NN.numerical_gradient_no_batch(f, x)
            
    #         return grad


    # def numerical_gradient(self,x,t):
    #     loss_W = lambda W: self.loss(x, t)   #람다 정규식> lambda 변수: return 식
    #     grads = {}
    #     grads['w1'] = NN.gradient(loss_W, self.params['w1'])
    #     grads['b1'] = NN.gradient(loss_W, self.params['b1'])
    #     grads['w2'] = NN.gradient(loss_W, self.params['w2'])
    #     grads['b2'] = NN.gradient(loss_W, self.params['b2'])
    #     #print(grads['w1'])
    #     return grads

    # def accuracy(self, x, t):
    #     y = self.predict(x)
    #     y = np.argmax(y) 

    #     # if t.ndim != 1 :   
    #     #     t = np.argmax(t, axis=1)
    #     t = np.argmax(t)
    #     #accuracy = np.sum(y == t) / float(x.shape[0])   

    #     if y==t :
    #         return 1
    #     else:
    #         return 0    
################################수치 미분법 코드###################
    def nomalize(self,x):                   #정규화 코드
        min=np.min(x,axis=0)
        max=np.max(x,axis=0)
        
        for i,val in enumerate(x):
            x[i]=(x[i]-min)/(max-min)

        return x        


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        # 가중치 초기화

        NN.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)/np.sqrt(hidden_size)   
        NN.params['b1'] = np.zeros(hidden_size)
        NN.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size) /np.sqrt(hidden_size)   
        NN.params['b2'] = np.zeros(output_size)

        # 계층 생성
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(NN.params['W1'], NN.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(NN.params['W2'], NN.params['b2'])

        self.lastLayer = SoftmaxWithLoss()
        
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        
        return x
        
    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        if y.ndim != 1 : y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        print('예측값: '+str(np.argmax(y)))

        if np.argmax(y)==np.argmax(t):
            return 1
        else:
            return 0    
    
    def accuracy_woldcup(self, x):
        y = self.predict(x)
        if y.ndim != 1 : y = np.argmax(y, axis=1)
        print('월드컵: '+str(np.argmax(y)))       
        
    # x : 입력 데이터, t : 정답 레이블
        
    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        return grads     

class Relu:
    def __init__(self):
        self.mask = None   # mask 배열의 원소가 True인 곳에는 상류에서 전파된 dout을 0으로 설정

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
        self.original_x_shape = None

    def forward(self, x):
        self.original_x_shape = x.shape
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out
        
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T.reshape(self.x.T.shape[0],1), dout.reshape(1,dout.shape[0]))
        self.db = np.sum(dout, axis=0)
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None # 손실함수
        self.y = None    # softmax의 출력
        self.t = None    # 정답 레이블(원-핫 인코딩 형태)
        
    def forward(self, x, t):
        self.t = t
        self.y = SoftmaxWithLoss.softmax(x)
        self.loss = SoftmaxWithLoss.cross_entropy_error(self.y, self.t)
        NN.plt_X.append(self.loss)
        print('loss: '+str(self.loss))
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 정답 레이블이 원-핫 인코딩 형태일 때
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx

    
    def softmax(x):
        
        if x.ndim == 2:
            x = x.T
            x = x - np.max(x, axis=0)
            y = np.exp(x) / np.sum(np.exp(x), axis=0)
            
            return y.T
            
        x = x - np.max(x)	# Overflow 대책
        
        return np.exp(x) / np.sum(np.exp(x))
        
    def cross_entropy_error(y, t):
        
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)
            
        # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 변환
        if t.size == y.size:
            t = t.argmax(axis=1)
            
        batch_size = y.shape[0]
        
        return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

