import sys, os 
sys.path.append(os.pardir)
import numpy as np 
from PIL import Image 
import pandas as pd

csv_data=pd.read_csv('./FIFAallMatchBoxData.csv')
print(csv_data.shape)


