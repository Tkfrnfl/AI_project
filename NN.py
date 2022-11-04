from cmath import nan
from math import sqrt
import math
import numpy as np
from PIL import Image 
from random import random
import matplotlib.pyplot as plt

class LR():
    learning_rate = 0.00001
    epochs = 500

    cost_list=[[0]*1000 for _ in range(10)]
    correct=0
    sum=0

    w1=np.random.rand(784,10)    # 가중치와 편향 초기화
    b1=np.random.rand(10)
