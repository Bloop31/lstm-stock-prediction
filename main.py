import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_loader import load_data
from src.preprocess import scale_data
from src.create_sequences import create_sequences
from src.model import build_model

data=load_data()

scaled_data,scaler=scale_data(data)

X,y=create_sequences(scaled_data)

split=int(len(X)*0.8)

X_train=X[:split]
X_test=X[split:]

y_train=y[:split]
y_test=y[split:]

model=build_model()

model.fit(
    X_train,y_train,epochs=10,batch_size=32
)
model.save("models/lstm_model.h5")

predictions=model.predict(X_test)

predictions=scaler.inverse_transform(predictions)
y_test_actual=scaler.inverse_transform(y_test)


plt.plot(y_test_actual,label="Actual Price")
plt.plot(predictions,label="Predicted Price")
plt.legend()
plt.savefig("results/prediction_plot.png")
plt.show()