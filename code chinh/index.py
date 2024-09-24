

import pandas as pd
from sklearn.model_selection import train_test_split

# Đọc dữ liệu từ file CSV
file_path = r'/Data/Advertising-Budget-and-Sales.csv'
df = pd.read_csv(file_path)

# Xóa cột số thứ tự
df = df.drop(columns=['Unnamed: 0'])

# Sao chép DataFrame để làm việc
df1 = df.copy()

# Thay đổi tên cột
df1.columns = [i.replace(' ', '_') for i in df1.columns]

# Định nghĩa biến mục tiêu
target = 'Sales_($)'

# Tách dữ liệu thành đặc trưng và mục tiêu
X = df1.drop([target], axis=1)
Y = df1[target]

# Loại bỏ các giá trị ngoại lai
features1 = X.columns  # Xử lý tất cả các cột

for i in features1:
    if X[i].dtype in ['int64', 'float64']:  # Chỉ xử lý các cột số
        Q1 = X[i].quantile(0.25)
        Q3 = X[i].quantile(0.75)
        IQR = Q3 - Q1
        X = X[X[i] <= (Q3 + (1.5 * IQR))]
        X = X[X[i] >= (Q1 - (1.5 * IQR))]
        Y = Y[X.index]  # Đồng bộ hóa với Y
        X.reset_index(drop=True, inplace=True)
        Y.reset_index(drop=True, inplace=True)

# Phân chia dữ liệu thành tập huấn luyện (80%) và tập còn lại (20%)
Train_X, Temp_X, Train_Y, Temp_Y = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=100)

# Phân chia tập còn lại thành tập validation (50%) và tập kiểm tra (50%)
Validation_X, Test_X, Validation_Y, Test_Y = train_test_split(Temp_X, Temp_Y, train_size=0.5, test_size=0.5, random_state=100)

# Kết hợp X và Y cho mỗi tập dữ liệu
Train_data = pd.concat([Train_X, Train_Y], axis=1)
Validation_data = pd.concat([Validation_X, Validation_Y], axis=1)
Test_data = pd.concat([Test_X, Test_Y], axis=1)

# Lưu các tập dữ liệu vào các file CSV
Train_data.to_csv(r'/Data/Train_data.csv', index=False)
Validation_data.to_csv(r'/Data/Validation_data.csv', index=False)
Test_data.to_csv(r'/Data/Test_data.csv', index=False)

print("Các file dữ liệu đã được lưu thành công!")
