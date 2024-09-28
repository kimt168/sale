
#CODE này ra luôn 3 mô hình 

# from flask import Flask, render_template, request, jsonify
# import joblib
# import pandas as pd

# app = Flask(__name__)

# # Tải các mô hình đã lưu
# linear_model = joblib.load(r'C:/Users\admin\Desktop\Học may\/Data/linear_regression_model.pkl')
# ridge_model = joblib.load(r'C:\Users\admin\Desktop\Học may\/Data/ridge_regression_model.pkl')
# mlp_model = joblib.load(r'C:\Users\admin\Desktop\Học may\/Data/mlp_regression_model.pkl')

# # Hàm dự đoán
# def predict(model, tv, radio, newspaper):
#     input_data = pd.DataFrame([[tv, radio, newspaper]], columns=['TV_Ad_Budget_($)', 'Radio_Ad_Budget_($)', 'Newspaper_Ad_Budget_($)'])
#     prediction = model.predict(input_data)
#     return prediction[0]

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict_sales():
#     try:
#         data = request.get_json()
#         tv = float(data['tv'])
#         radio = float(data['radio'])
#         newspaper = float(data['newspaper'])
        
#         linear_pred = predict(linear_model, tv, radio, newspaper)
#         ridge_pred = predict(ridge_model, tv, radio, newspaper)
#         mlp_pred = predict(mlp_model, tv, radio, newspaper)
        
#         return jsonify({
#             'linear': float(linear_pre2),
#             'ridge': float(ridge_pred),
#             'mlp': float(mlp_pred, #         })
#     except Exception as e:
#         return jsonify({'error': str(e)})

# if __name__ == '__main__':
#     app.run(debug=True)











# FINAL

# from flask import Flask, render_template, request, jsonify
# import joblib
# import pandas as pd

# app = Flask(__name__)

# # Tải các mô hình đã lưu
# linear_model = joblib.load(r'C:\Users\admin\Desktop\Học may\/Data/linear_regression_model.pkl')
# ridge_model = joblib.load(r'C:\Users\admin\Desktop\Học may\/Data/ridge_regression_model.pkl')
# mlp_model = joblib.load(r'C:\Users\admin\Desktop\Học may\/Data/mlp_regression_model.pkl')

# # Hàm dự đoán
# def predict(model, tv, radio, newspaper):
#     input_data = pd.DataFrame([[tv, radio, newspaper]], columns=['TV_Ad_Budget_($)', 'Radio_Ad_Budget_($)', 'Newspaper_Ad_Budget_($)'])
#     prediction = model.predict(input_data)
#     return prediction[0]

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict_sales():
#     try:
#         data = request.get_json()
#         tv = float(data['tv'])
#         radio = float(data['radio'])
#         newspaper = float(data['newspaper'])
#         model_choice = data['model']

#         # Chọn mô hình dự đoán
#         if model_choice == 'linear':
#             model = linear_model
#         elif model_choice == 'ridge':
#             model = ridge_model
#         elif model_choice == 'mlp':
#             model = mlp_model
#         else:
#             return jsonify({'error': 'Invalid model choice'})

#         prediction = predict(model, tv, radio, newspaper)
        
#         return jsonify({
#             'model': model_choice,
#             'prediction': float(prediction)
#         })
#     except Exception as e:
#         return jsonify({'error': str(e)})

# if __name__ == '__main__':
#     app.run(debug=True)










# from flask import Flask, render_template, request, jsonify, send_from_directory
# import joblib
# import pandas as pd
# import os

# app = Flask(__name__)

# # Thư mục chứa các hình ảnh đồ thị
# PLOTS_DIR = r'C:\Users\admin\Desktop\Học may\/Data/

# # Tải các mô hình đã lưu
# linear_model = joblib.load(r'C:\Users\admin\Desktop\Học may\/Data/linear_regression_model.pkl')
# ridge_model = joblib.load(r'C:\Users\admin\Desktop\Học may\/Data/ridge_regression_model.pkl')
# mlp_model = joblib.load(r'C:\Users\admin\Desktop\Học may\/Data/mlp_regression_model.pkl')

# # Hàm dự đoán
# def predict(model, tv, radio, newspaper):
#     input_data = pd.DataFrame([[tv, radio, newspaper]], columns=['TV_Ad_Budget_($)', 'Radio_Ad_Budget_($)', 'Newspaper_Ad_Budget_($)'])
#     prediction = model.predict(input_data)
#     return prediction[0]

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict_sales():
#     try:
#         data = request.get_json()
#         tv = float(data['tv'])
#         radio = float(data['radio'])
#         newspaper = float(data['newspaper'])
#         model_choice = data['model']

#         # Chọn mô hình dự đoán
#         if model_choice == 'linear':
#             model = linear_model
#             plot_filename = 'linear_regression_plot.png'
#         elif model_choice == 'ridge':
#             model = ridge_model
#             plot_filename = 'ridge_regression_plot.png'
#         elif model_choice == 'mlp':
#             model = mlp_model
#             plot_filename = 'mlp_regression_plot.png'
#         else:
#             return jsonify({'error': 'Invalid model choice'})

#         prediction = predict(model, tv, radio, newspaper)
        
#         return jsonify({
#             'model': model_choice,
#             'prediction': float(prediction),
#             'plot_url': f'/plots/{plot_filename}'
#         })
#     except Exception as e:
#         return jsonify({'error': str(e)})

# @app.route('/plots/<filename>')
# def get_plot(filename):
#     return send_from_directory(PLOTS_DIR, filename)

# if __name__ == '__main__':
#     app.run(debug=True)





# from flask import Flask, render_template, request, jsonify, send_from_directory
# import joblib
# import pandas as pd
# import numpy as np
# from sklearn.metrics import r2_score, mean_squared_error
# import os

# app = Flask(__name__)

# # Thư mục chứa các hình ảnh đồ thị
# PLOTS_DIR = r'C:\Users\admin\Desktop\Học may\/Data/

# # Tải các mô hình đã lưu
# linear_model = joblib.load(r'C:\Users\admin\Desktop\Học may\/Data/linear_regression_model.pkl')
# ridge_model = joblib.load(r'C:\Users\admin\Desktop\Học may\/Data/ridge_regression_model.pkl')
# mlp_model = joblib.load(r'C:\Users\admin\Desktop\Học may\/Data/mlp_regression_model.pkl')

# # Đọc dữ liệu để đánh giá mô hình
# Train_X_std = pd.read_csv(r'C:\Users\admin\Desktop\Học may\/Data/Train_X_std.csv')
# Train_Y = pd.read_csv(r'C:\Users\admin\Desktop\Học may\/Data/Train_Y.csv').values.ravel()
# Val_X_std = pd.read_csv(r'C:\Users\admin\Desktop\Học may\/Data/Val_X_std.csv')
# Val_Y = pd.read_csv(r'C:\Users\admin\Desktop\Học may\/Data/Val_Y.csv').values.ravel()
# Test_X_std = pd.read_csv(r'C:\Users\admin\Desktop\Học may\/Data/Test_X_std.csv')
# Test_Y = pd.read_csv(r'C:\Users\admin\Desktop\Học may\/Data/Test_Y.csv').values.ravel()

# # Hàm đánh giá mô hình
# def evaluate_model(model, X_train, Y_train, X_val, Y_val, X_test, Y_test):
#     # Dự đoán trên tập huấn luyện, tập xác thực và tập kiểm tra
#     train_preds = model.predict(X_train)
#     val_preds = model.predict(X_val)
#     test_preds = model.predict(X_test)

#     # Tính toán các chỉ số đánh giá
#     train_r2 = r2_score(Y_train, train_preds)
#     val_r2 = r2_score(Y_val, val_preds)
#     test_r2 = r2_score(Y_test, test_preds)
#     train_mse = mean_squared_error(Y_train, train_preds)
#     val_mse = mean_squared_error(Y_val, val_preds)
#     test_mse = mean_squared_error(Y_test, test_preds)
#     train_rmse = np.sqrt(train_mse)
#     val_rmse = np.sqrt(val_mse)
#     test_rmse = np.sqrt(test_mse)

#     return {
#         'train_r2': train_r2,
#         'val_r2': val_r2,
#         'test_r2': test_r2,
#         'train_mse': train_mse,
#         'val_mse': val_mse,
#         'test_mse': test_mse,
#         'train_rmse': train_rmse,
#         'val_rmse': val_rmse,
#         'test_rmse': test_rmse
#     }

# # Hàm dự đoán
# def predict(model, tv, radio, newspaper):
#     input_data = pd.DataFrame([[tv, radio, newspaper]], columns=['TV_Ad_Budget_($)', 'Radio_Ad_Budget_($)', 'Newspaper_Ad_Budget_($)'])
#     prediction = model.predict(input_data)[0]
#     metrics = evaluate_model(model, Train_X_std, Train_Y, Val_X_std, Val_Y, Test_X_std, Test_Y)
#     return prediction, metrics

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict_sales():
#     try:
#         data = request.get_json()
#         tv = float(data['tv'])
#         radio = float(data['radio'])
#         newspaper = float(data['newspaper'])
#         model_choice = data['model']

#         # Chọn mô hình dự đoán
#         if model_choice == 'linear':
#             model = linear_model
#             plot_filename = 'linear_regression_plot.png'
#         elif model_choice == 'ridge':
#             model = ridge_model
#             plot_filename = 'ridge_regression_plot.png'
#         elif model_choice == 'mlp':
#             model = mlp_model
#             plot_filename = 'mlp_regression_plot.png'
#         else:
#             return jsonify({'error': 'Invalid model choice'})

#         prediction, metrics = predict(model, tv, radio, newspaper)

#         return jsonify({
#             'model': model_choice,
#             'prediction': float(prediction),
#             'plot_url': f'/plots/{plot_filename}',
#             'metrics': metrics
#         })
#     except Exception as e:
#         return jsonify({'error': str(e)})

# @app.route('/plots/<filename>')
# def get_plot(filename):
#     return send_from_directory(PLOTS_DIR, filename)

# if __name__ == '__main__':
#     app.run(debug=True)










from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV

app = Flask(__name__)

# Tải các mô hình đã lưu
linear_model = joblib.load(r'./Data/linear_regression_model.pkl')
ridge_model = joblib.load(r'./Data/ridge_regression_model.pkl')
mlp_model = joblib.load(r'./Data/mlp_regression_model.pkl')
stacking_model = joblib.load(r'meta_model.pkl')


# Đọc dữ liệu để đánh giá mô hình
Train_X_std = pd.read_csv(r'./Data/Train_X_std.csv')
Train_Y = pd.read_csv(r'./Data/Train_Y.csv').values.ravel()
Val_X_std = pd.read_csv(r'./Data/Val_X_std.csv')
Val_Y = pd.read_csv(r'./Data/Val_Y.csv').values.ravel()
Test_X_std = pd.read_csv(r'./Data/Test_X_std.csv')
Test_Y = pd.read_csv(r'./Data/Test_Y.csv').values.ravel()


def stacking_evaluated():
        # Step 1: Load the data
    train_X = pd.read_csv('./TrainingData/Train_X_std.csv')
    train_Y = pd.read_csv('./TrainingData/Train_Y.csv')
    val_X = pd.read_csv('./TrainingData/Val_X_std.csv')
    val_Y = pd.read_csv('./TrainingData/Val_Y.csv')
    test_X = pd.read_csv('./TrainingData/Test_X_std.csv')
    test_Y = pd.read_csv('./TrainingData/Test_Y.csv')

    # Step 2: Load pre-trained models
    linear_model_train = joblib.load('./TrainingData/linear_regression_model.pkl')
    mlp_model_train = joblib.load('./TrainingData/mlp_regression_model.pkl')
    ridge_model_train = joblib.load('./TrainingData/ridge_regression_model2.pkl')

    # Step 3: Create first-level predictions
    train_pred_linear = linear_model_train.predict(train_X)
    train_pred_mlp = mlp_model_train.predict(train_X)
    train_pred_ridge = ridge_model_train.predict(train_X)

    val_pred_linear = linear_model_train.predict(val_X)
    val_pred_mlp = mlp_model_train.predict(val_X)
    val_pred_ridge = ridge_model_train.predict(val_X)

    test_pred_linear = linear_model_train.predict(test_X)
    test_pred_mlp = mlp_model_train.predict(test_X)
    test_pred_ridge = ridge_model_train.predict(test_X)

    # Stack predictions for meta-model
    train_meta_X = np.column_stack((train_pred_linear, train_pred_mlp, train_pred_ridge))
    val_meta_X = np.column_stack((val_pred_linear, val_pred_mlp, val_pred_ridge))
    test_meta_X = np.column_stack((test_pred_linear, test_pred_mlp, test_pred_ridge))

    # Step 4: Use GridSearchCV to optimize Ridge Regression
    param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0]}
    ridge_cv = GridSearchCV(Ridge(), param_grid, cv=5)
    ridge_cv.fit(train_meta_X, train_Y)

    # Use the best model from GridSearchCV
    meta_model = ridge_cv.best_estimator_

    # Step 5: Evaluate the meta-model on validation set
    val_meta_pred = meta_model.predict(val_meta_X)
    val_mse = mean_squared_error(val_Y, val_meta_pred)
    val_r2 = r2_score(val_Y, val_meta_pred)

    print(f'Validation MSE of Stacked Model: {val_mse}')
    print(f'Validation R-squared of Stacked Model: {val_r2}')

    # Step 6: Make final predictions on the test set
    test_meta_pred = meta_model.predict(test_meta_X)
    test_mse = mean_squared_error(test_Y, test_meta_pred)
    test_r2 = r2_score(test_Y, test_meta_pred)

    print(f'Test MSE of Stacked Model: {test_mse}')
    print(f'Test R-squared of Stacked Model: {test_r2}')

    # Step 7: Perform cross-validation for generalization check
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(meta_model, train_meta_X, train_Y, cv=5, scoring='r2')
    # Step 8: Calculate and print the evaluation metrics
    # Training evaluation
    train_meta_pred = meta_model.predict(train_meta_X)
    train_r2 = r2_score(train_Y, train_meta_pred)
    train_mse = mean_squared_error(train_Y, train_meta_pred)
    train_rmse = np.sqrt(train_mse)
    val_rmse = np.sqrt(val_mse)
    test_rmse = np.sqrt(test_mse)
    return {
        'train_r2': train_r2,
        'val_r2': val_r2,
        'test_r2': test_r2,
        'train_mse': train_mse,
        'val_mse': val_mse,
        'test_mse': test_mse,
        'train_rmse': train_rmse,
        'val_rmse': val_rmse,
        'test_rmse': test_rmse
    }


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

    return {
        'train_r2': train_r2,
        'val_r2': val_r2,
        'test_r2': test_r2,
        'train_mse': train_mse,
        'val_mse': val_mse,
        'test_mse': test_mse,
        'train_rmse': train_rmse,
        'val_rmse': val_rmse,
        'test_rmse': test_rmse
    }

# Hàm dự đoán
def predict(model, tv, radio, newspaper):
    if(model == stacking_model):
         # Tạo mảng dữ liệu từ các giá trị đầu vào
        input_data = np.array([[tv, radio, newspaper]])
    
        train_X = pd.read_csv('./Data/Train_X_std.csv')

        # Giả sử các cột trong train_X có tên giống như ['feature1', 'feature2', 'feature3']
        input_data_df = pd.DataFrame(input_data, columns=train_X.columns)

        # # Step 2: Load pre-trained models
        # linear_model = joblib.load('linear_regression_model.pkl')
        # mlp_model = joblib.load('mlp_regression_model.pkl')
        # ridge_model = joblib.load('ridge_regression_model2.pkl')

        # Bước 1: Dự đoán từ các mô hình cấp 1 (Linear, MLP, Ridge)
        pred_linear = linear_model.predict(input_data_df)
        pred_mlp = mlp_model.predict(input_data_df)
        pred_ridge = ridge_model.predict(input_data_df)
        # Bước 2: Stack các dự đoán từ mô hình cấp 1
        meta_input = np.column_stack((pred_linear, pred_mlp, pred_ridge))
        
        model = joblib.load('stacked_meta_model_best_alpha.pkl')
        prediction = model.predict(meta_input) 
        metrics = stacking_evaluated()
    else:
        input_data = pd.DataFrame([[tv, radio, newspaper]], columns=['TV_Ad_Budget_($)', 'Radio_Ad_Budget_($)', 'Newspaper_Ad_Budget_($)'])
        prediction = model.predict(input_data)[0]
        metrics = evaluate_model(model, Train_X_std, Train_Y, Val_X_std, Val_Y, Test_X_std, Test_Y)
    return prediction, metrics

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_sales():
    try:
        data = request.get_json()
        tv = float(data['tv'])
        radio = float(data['radio'])
        newspaper = float(data['newspaper'])
        model_choice = data['model']

        # Chọn mô hình dự đoán
        if model_choice == 'linear':
            model = linear_model
        elif model_choice == 'ridge':
            model = ridge_model
        elif model_choice == 'neural_network':
            model = mlp_model
        elif model_choice == 'stacking':
            model = stacking_model
        else:
            return jsonify({'error': 'Invalid model choice'})

        prediction, metrics = predict(model, tv, radio, newspaper)

        return jsonify({
            'model': model_choice,
            'prediction': float(prediction),
            # 'metrics': metrics
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
