import numpy as np
def create_sequences(data,window_size=60):
    X=[]
    y=[]
    for i in range(window_size,len(data)):
        X.append(data[i-window_size:i])
        y.append(data[i])
    X=np.array(X)
    y=np.array(y)
    return X,y