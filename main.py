import sys, os 
sys.path.append(os.pardir)
import numpy as np 
from PIL import Image 
import pandas as pd
import csv

# csv_data=pd.read_csv('./FIFAallMatchBoxData.csv')

#csv 데이터를 각 나라별 데이터로 재가공
#경기에 긍정적 영향을주는것, 부정적 영향을 주는걸로 2차분류

county_list=[]

f=open('./FIFAallMatchBoxData.csv',"r")
csv_data=csv.reader(f)

for i in csv_data:
    





