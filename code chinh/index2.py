
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import matplotlib.pyplot as plt

# Đọc dữ liệu đã được xử lý và tiêu chuẩn hóa
Train_X_std = pd.read_csv(r'C:\Users\admin\Desktop\Học may\sale\Data\Train_X_std.csv')
Train_Y = pd.read_csv(r'C:\Users\admin\Desktop\Học may\sale\Data\Train_Y.csv').values.ravel()
Val_X_std = pd.read_csv(r'C:\Users\admin\Desktop\Học may\sale\Data\Val_X_std.csv')
Val_Y = pd.read_csv(r'C:\Users\admin\Desktop\Học may\sale\Data\Val_Y.csv').values.ravel()
Test_X_std = pd.read_csv(r'C:\Users\admin\Desktop\Học may\sale\Data\Test_X_std.csv')
Test_Y = pd.read_csv(r'C:\Users\admin\Desktop\Học may\sale\Data\Test_Y.csv').values.ravel()

# Hàm đánh giá mô hình
def evaluate_model(model, X_train, Y_train, X_val, Y_val, X_test, Y_test):
    # Dự đoán trên tập huấn luyện, tập xác thực và tập kiểm tra
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    test_preds = model.predict(X_test)

    # Tính toán các chỉ số đánh giá
    train_r2 = r2_score(Y_train, train_preds)
    val_r2 = r2_score(Y_val, val_preds)
    test_r2 = r2_score(Y_test, test_preds)
    train_mse = mean_squared_error(Y_train, train_preds)
    val_mse = mean_squared_error(Y_val, val_preds)
    test_mse = mean_squared_error(Y_test, test_preds)
    train_rmse = np.sqrt(train_mse)
    val_rmse = np.sqrt(val_mse)
    test_rmse = np.sqrt(test_mse)

    print(f'R2 Score on Training Set: {train_r2:.4f}')
    print(f'R2 Score on Validation Set: {val_r2:.4f}')
    print(f'R2 Score on Testing Set: {test_r2:.4f}')
    print(f'MSE on Training Set: {train_mse:.4f}')
    print(f'MSE on Validation Set: {val_mse:.4f}')
    print(f'MSE on Testing Set: {test_mse:.4f}')
    print(f'RMSE on Training Set: {train_rmse:.4f}')
    print(f'RMSE on Validation Set: {val_rmse:.4f}')
    print(f'RMSE on Testing Set: {test_rmse:.4f}')

    return train_r2, val_r2, test_r2, train_mse, val_mse, test_mse, train_rmse, val_rmse, test_rmse

# Hàm vẽ và lưu đồ thị
def plot_predictions(model, X, Y, title, filename):
    predictions = model.predict(X)
    plt.figure(figsize=(10, 6))
    plt.scatter(Y, predictions, alpha=0.5)
    plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'r--')
    plt.title(title)
    plt.xlabel('y_test')
    plt.ylabel('y_pred')
    plt.savefig(filename)
    plt.close()


# 1. Linear Regression
print("Linear Regression:")
lin_reg = LinearRegression()
lin_reg.fit(Train_X_std, Train_Y)
evaluate_model(lin_reg, Train_X_std, Train_Y, Val_X_std, Val_Y, Test_X_std, Test_Y)
joblib.dump(lin_reg, r'/Data/linear_regression_model.pkl')
plot_predictions(lin_reg, Train_X_std, Train_Y, 'Linear Regression: Test vs Prediction', r'Data/linear_regression_plot.png')

# 2. Ridge Regression
print("\nRidge Regression:")
ridge_reg = Ridge(alpha=1.0)
ridge_reg.fit(Train_X_std, Train_Y)
evaluate_model(ridge_reg, Train_X_std, Train_Y, Val_X_std, Val_Y, Test_X_std, Test_Y)
joblib.dump(ridge_reg, r'/Data/ridge_regression_model.pkl')
plot_predictions(ridge_reg, Train_X_std, Train_Y, 'Ridge Regression: Test vs Prediction', r'/Data/ridge_regression_plot.png')

# 3. Neural Network (MLPRegressor)
print("\nNeural Network (MLPRegressor):")
mlp_reg = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=1)
mlp_reg.fit(Train_X_std, Train_Y)
evaluate_model(mlp_reg, Train_X_std, Train_Y, Val_X_std, Val_Y, Test_X_std, Test_Y)
joblib.dump(mlp_reg, r'/Data/mlp_regression_model.pkl')
plot_predictions(mlp_reg, Train_X_std, Train_Y, 'Neural Network (MLPRegressor): Test vs Prediction', r'/Data/mlp_regression_plot.png')

print("\nMô hình đã được huấn luyện và lưu thành công.")
