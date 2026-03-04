from sklearn.linear_model import LinearRegression
def train_baseline(X_train,y_train):
    X_train=X_train.reshape(X_train.shape[0],-1)
    model=LinearRegression()
    model.fit(X_train,y_train)
    return model