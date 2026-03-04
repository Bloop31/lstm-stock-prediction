import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_loader import load_data
from src.preprocess import scale_data
from src.create_sequences import create_sequences
from src.model import build_model
from sklearn.metrics import mean_absolute_error,mean_squared_error
from src.baseline import train_baseline

data=load_data()

scaled_data,scaler=scale_data(data)

X,y=create_sequences(scaled_data)

split=int(len(X)*0.8)

X_train=X[:split]
X_test=X[split:]

y_train=y[:split]
y_test=y[split:]
# Train baseline model
baseline_model = train_baseline(X_train, y_train)
# Baseline predictions
X_test_flat = X_test.reshape(X_test.shape[0], -1)
baseline_preds = baseline_model.predict(X_test_flat)

baseline_preds = scaler.inverse_transform(baseline_preds.reshape(-1,1))

model=build_model()

model.fit(
    X_train,y_train,epochs=10,batch_size=32
)
model.save("models/lstm_model.h5")

predictions=model.predict(X_test)

predictions=scaler.inverse_transform(predictions)
y_test_actual=scaler.inverse_transform(y_test)

rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
mae = mean_absolute_error(y_test_actual, predictions)

baseline_rmse = np.sqrt(mean_squared_error(y_test_actual, baseline_preds))
baseline_mae = mean_absolute_error(y_test_actual, baseline_preds)


print("LSTM RMSE:", rmse)
print("LSTM MAE:", mae)

print("Baseline RMSE:", baseline_rmse)
print("Baseline MAE:", baseline_mae)

plt.plot(y_test_actual, label="Actual Price")
plt.plot(predictions, label="LSTM Prediction")
plt.plot(baseline_preds, label="Linear Regression Baseline")
plt.legend()
plt.savefig("results/prediction_plot.png")
plt.show()