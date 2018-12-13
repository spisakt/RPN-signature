import numpy as np

def bmi(height, weight):
    return weight.astype(float)/np.square(height.astype(float)/100)



