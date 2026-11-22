#import hàm để lấy dataset từ UCI
from ucimlrepo import fetch_ucirepo

# fetch dataset
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

# data (as pandas dataframes)
X = breast_cancer_wisconsin_diagnostic.data.features
y = breast_cancer_wisconsin_diagnostic.data.targets

# thông tin tổng quát của dữ liệu
print(breast_cancer_wisconsin_diagnostic.metadata)

# thông tin của từng biến
print(breast_cancer_wisconsin_diagnostic.variables)

#2.5 KHOẢNG GIÁ TRỊ CỦA TỪNG BIẾN
# Giá trị min, max của tất cả biến
X.min()
X.max()
# Tạo bảng đầy đủ
range_df = X.agg(['min', 'max']).T
range_df['range'] = range_df['max'] - range_df['min']
print(range_df)
X.agg(['min', 'max'])
print(X.agg(['min', 'max']))

#3.1
import pandas as pd

# Thiết lập hiển thị tất cả các cột (không bị ...)
pd.set_option('display.max_columns', None)
# Thiết lập hiển thị tất cả các dòng
pd.set_option('display.max_rows', None)

# Sau đó chạy lệnh thống kê mô tả
print(X.describe().T)

#3.2
#Thống kê mô tả biến mục tiêu
y['Diagnosis'].value_counts()
y['Diagnosis'].value_counts(normalize=True)
print(y['Diagnosis'].value_counts())
print(y['Diagnosis'].value_counts(normalize=True))
import matplotlib.pyplot as plt
import seaborn as sns

# Chọn 10 biến đầu (mean features)
features = X.columns[:10]   

# Tạo figure
plt.figure(figsize=(15, 8))

# Vẽ boxplot cho từng biến
for i, col in enumerate(features):
    plt.subplot(2, 5, i+1)
    sns.boxplot(y=X[col])
    plt.title(col)

plt.tight_layout()
plt.show()

#kiểm tra tính tương quan + vẽ biểu đồ
import pandas as pd

corr_matrix = X.corr()
import numpy as np

corr_matrix = X.corr()
print(X.corr().head)
import seaborn as sns
import matplotlib.pyplot as plt

# Chọn các biến mean (10 biến đầu)
mean_features = X.columns[:10]   

# Tính ma trận tương quan
corr_matrix = X[mean_features].corr()

# Vẽ heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_matrix,
    annot=True,     
    cmap='coolwarm',   
    fmt=".2f",         
    linewidths=0.5
)

plt.title("Correlation Heatmap of Mean Features")
plt.show()

# ===== DATA CLEANING =====
df = X.copy()
df['Diagnosis'] = y

# Xóa cột ID nếu có
#df = df.drop('ID', axis=1)
# Gộp dữ liệu
df = X.copy()
df['Diagnosis'] = y

# Kiểm tra duplicate
print("Duplicate rows:", df.duplicated().sum())

# Tách lại
X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']
# Kiểm tra số lượng giá trị rỗng theo từng cột
print(X.isnull().sum())

# Kiểm tra toàn bộ dataset có missing không
print("Tổng số missing:", X.isnull().sum().sum())
print(df.columns)


## 2.1.4 Kiểm tra tính hợp lý của dữ liệu
# Kiểm tra giá trị âm trong features (các đặc trưng đều là kích thước tế bào, không thể âm)
import numpy as np
negative_values = (X < 0).sum().sum()
print(f"Số giá trị âm: {negative_values}")

# Kiểm tra giá trị vô cùng (inf)
infinite_values = np.isinf(X).sum().sum()
print(f"Số giá trị vô cùng: {infinite_values}")

# Kiểm tra outlier bằng IQR (ví dụ với feature 'radius_mean')
Q1 = X['radius1'].quantile(0.25)
Q3 = X['radius1'].quantile(0.75)
IQR = Q3 - Q1
outliers = ((X['radius1'] < (Q1 - 1.5 * IQR)) | (X['radius1'] > (Q3 + 1.5 * IQR))).sum()
print(f"Số outlier trong radius1 (radius_mean): {outliers}")
#2.1.5 trực quan hóa outlier
import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x=X['radius1'])
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

from sklearn.model_selection import train_test_split

# Chia dữ liệu: 80% train - 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% test
    random_state=42,    # đảm bảo tái lập kết quả
    stratify=y          # giữ tỷ lệ lớp 
)

# Kiểm tra kích thước
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

# Đếm số lượng từng class
print("Train:")
print(y_train.value_counts())

print("\nTest:")
print(y_test.value_counts())

# Tính tỷ lệ (%)
print("\nTrain (%):")
print(y_train.value_counts(normalize=True))

print("\nTest (%):")
print(y_test.value_counts(normalize=True))

# 2. Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)   # học từ train
X_test = scaler.transform(X_test)         # áp dụng cho test
import pandas as pd

X_train_df = pd.DataFrame(X_train, columns=X.columns)

print(X_train_df.mean())  # ≈ 0
print(X_train_df.std())   # ≈ 1

#=======Gausian NB=============
import sklearn.naive_bayes
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
model_nb = sklearn.naive_bayes.GaussianNB()

# Train model
model_nb.fit(X_train, y_train)
#dự đoán
y_pred_nb = model_nb.predict(X_test)
print(y_pred_nb)
acc_nb = accuracy_score(y_test, y_pred_nb)
print("Accuracy:", acc_nb)
# Precision, Recall, F1 (vì là binary B/M nên cần chỉ định pos_label)
precision = precision_score(y_test, y_pred_nb, pos_label='M')
recall = recall_score(y_test, y_pred_nb, pos_label='M')
f1 = f1_score(y_test, y_pred_nb, pos_label='M')
print("Accuracy:", acc_nb)
print("Precision (M):", precision)
print("Recall (M):", recall)
print("F1-score (M):", f1)
#Confusion Matrix
cm = confusion_matrix(y_test, y_pred_nb)
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", classification_report(y_test, y_pred_nb))

print("\nTABLE I. GNB EVALUATION METRICS")
print(f"Accuracy (%):  {acc_nb*100:.2f}")
print(f"Precision (M) (%): {precision*100:.2f}")
print(f"Recall (M) (%):    {recall*100:.2f}")
print(f"F1-Score (M) (%):  {f1*100:.2f}")


#Trực quan hóa
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Gaussian NB")
plt.show()

#========== Logistic Regression =============
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score 
model_lr = LogisticRegression(max_iter=1000)

# Train
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)
# Accuracy
acc = accuracy_score(y_test, y_pred_lr)

# Precision, Recall, F1 (vì là binary B/M nên cần chỉ định pos_label)
precision = precision_score(y_test, y_pred_lr, pos_label='M')
recall = recall_score(y_test, y_pred_lr, pos_label='M')
f1 = f1_score(y_test, y_pred_lr, pos_label='M')

print("Accuracy:", acc)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

cm = confusion_matrix(y_test, y_pred_lr)
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", classification_report(y_test, y_pred_lr))
print("\nTABLE II. LR EVALUATION METRICS")
print(f"Accuracy (%):  {acc*100:.2f}")
print(f"Precision (%): {precision*100:.2f}")
print(f"Recall (%):    {recall*100:.2f}")
print(f"F1-Score (%):  {f1*100:.2f}")

#=========== SVM ==============
from sklearn.svm import SVC

# Khởi tạo model (kernel mặc định là RBF)
model_svm = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)

# Train
model_svm.fit(X_train, y_train)

# Predict
y_pred_svm = model_svm.predict(X_test)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Accuracy
acc_svm = accuracy_score(y_test, y_pred_svm)
# Precision, Recall, F1 (vì là binary B/M nên cần chỉ định pos_label)
precision = precision_score(y_test, y_pred_svm, pos_label='M')
recall = recall_score(y_test, y_pred_svm, pos_label='M')
f1 = f1_score(y_test, y_pred_svm, pos_label='M')

print("Accuracy:", acc_svm)
print("Precision:",precision)
print("Recall:", recall)
print("F1-score:", f1)
# Confusion Matrix
cm_svm = confusion_matrix(y_test, y_pred_svm)
print("Confusion Matrix:\n", cm_svm)

# Classification Report
print("Classification Report:\n", classification_report(y_test, y_pred_svm))
from sklearn.metrics import precision_score, recall_score, f1_score

print("\nTABLE III: SVM EVALUATION METRICS")
print(f"Accuracy (%):  {acc_svm*100:.2f}")
print(f"Precision (%): {precision*100:.2f}")
print(f"Recall (%):    {recall*100:.2f}")
print(f"F1-Score (%):  {f1*100:.2f}")

#============ KNN ==============
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Khởi tạo mô hình (k = 5 là mặc định phổ biến)
knn = KNeighborsClassifier(n_neighbors=5)

# Train model
knn.fit(X_train, y_train)

# Dự đoán
y_pred_knn = knn.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))
# Đánh giá
acc_knn = accuracy_score(y_test, y_pred_knn)
precision = precision_score(y_test, y_pred_knn, pos_label='M')
recall = recall_score(y_test, y_pred_knn, pos_label='M')
f1 = f1_score(y_test, y_pred_knn, pos_label='M')
print("Accuracy:", acc_knn)
print("Precision (M):", precision)
print("Recall (M):", recall)
print("F1-score (M):", f1)

print("Classification Report:\n", classification_report(y_test, y_pred_knn))
print("\nTABLE IV. KNN EVALUATION METRICS")
print(f"Accuracy (%):  {acc_knn*100:.2f}")
print(f"Precision (M) (%): {precision*100:.2f}")
print(f"Recall (M) (%):    {recall*100:.2f}")
print(f"F1-Score (M) (%):  {f1*100:.2f}")

k_values = range(1, 21)
accuracy_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy_scores.append(accuracy_score(y_test, y_pred))
# Vẽ biểu đồ
plt.plot(k_values, accuracy_scores, marker='o')
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.title("Choosing Optimal K")
plt.show()
## K=7 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Khởi tạo mô hình (k = 7)
knn1 = KNeighborsClassifier(n_neighbors=7)

# Train model
knn1.fit(X_train, y_train)

# Dự đoán
y_pred_knn7 = knn1.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))
# Đánh giá
acc_knn = accuracy_score(y_test, y_pred_knn)
precision = precision_score(y_test, y_pred_knn, pos_label='M')
recall = recall_score(y_test, y_pred_knn, pos_label='M')
f1 = f1_score(y_test, y_pred_knn, pos_label='M')
print("Accuracy:", acc_knn)
print("Precision (M):", precision)
print("Recall (M):", recall)
print("F1-score (M):", f1)

print("Classification Report:\n", classification_report(y_test, y_pred_knn))
print("\nTABLE IV. KNN EVALUATION METRICS")
print(f"Accuracy (%):  {acc_knn*100:.2f}")
print(f"Precision (M) (%): {precision*100:.2f}")
print(f"Recall (M) (%):    {recall*100:.2f}")
print(f"F1-Score (M) (%):  {f1*100:.2f}")

k_values = range(1, 21)
accuracy_scores = []
print((y_pred_knn == y_pred_knn7).sum())

#=========== RF================
from sklearn.ensemble import RandomForestClassifier

# Khởi tạo model
rf = RandomForestClassifier(
    n_estimators=100,      # số cây
    random_state=42
)

# Train
rf.fit(X_train, y_train)

# Dự đoán
y_pred_rf = rf.predict(X_test)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

# Accuracy
acc_rf = accuracy_score(y_test, y_pred_rf)

# Các chỉ số lớp M
precision = precision_score(y_test, y_pred_rf, pos_label='M')
recall = recall_score(y_test, y_pred_rf, pos_label='M')
f1 = f1_score(y_test, y_pred_rf, pos_label='M')

# In kết quả
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Accuracy:", acc_rf)
print("Precision (M):", precision)
print("Recall (M):", recall)
print("F1-score (M):", f1)

print("Classification Report:\n", classification_report(y_test, y_pred_rf))

print("\nTABLE V. RANDOM FOREST EVALUATION METRICS")
print(f"Accuracy (%):  {acc_rf*100:.2f}")
print(f"Precision (M) (%): {precision*100:.2f}")
print(f"Recall (M) (%):    {recall*100:.2f}")
print(f"F1-Score (M) (%):  {f1*100:.2f}")

#========== ROC-AUC ======================
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

# Chuyển y về dạng nhị phân (B=0, M=1)
y_test_bin = label_binarize(y_test, classes=['B', 'M']).ravel()

# Khởi tạo models
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

models = {
    "GaussianNB": GaussianNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(random_state=42)
}

plt.figure(figsize=(8,6))

for name, model in models.items():
    # Train
    model.fit(X_train, y_train)
    
    # Lấy xác suất lớp M (class 1)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # ROC
    fpr, tpr, _ = roc_curve(y_test_bin, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Plot
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")

# Đường random
plt.plot([0,1], [0,1], 'k--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid()

plt.show()

#===== AUC riêng =========== 
from sklearn.metrics import roc_auc_score

print("AUC Scores:")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test_bin, y_prob)
    
    print(f"{name}: {auc_score:.4f}")

#======= TUNING SVM ==========
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# Khởi tạo model
svm = SVC(kernel='rbf', probability=True)

# Grid tham số
param_grid = {
    'C': [0.1, 1, 10, 15, 20],
    'gamma': [0.0001, 0.001, 0.01, 0.1]
}

# GridSearchCV
grid = GridSearchCV(
    estimator=svm,
    param_grid=param_grid,
    cv=5,                 # 5-fold cross validation
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)

# Train
grid.fit(X_train, y_train)

# Kết quả tốt nhất
print("Best Parameters:", grid.best_params_)
print("Best CV Score:", grid.best_score_)
best_svm = grid.best_estimator_

# Predict
y_pred_svm = best_svm.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))
print("Classification Report:\n", classification_report(y_test, y_pred_svm))
import pandas as pd

results = pd.DataFrame(grid.cv_results_)
print(results[['param_C', 'param_gamma', 'mean_test_score']])

#========= TUNING RF ===============
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Khởi tạo model
rf = RandomForestClassifier(random_state=42)

# Grid tham số
param_grid = {
    'n_estimators': [100, 130, 150],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 3, 5],
    'min_samples_leaf': [1, 2, 3],
    'max_features': ['sqrt', 'log2', 0.5]
}

# Grid Search
grid_rf = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)

# Train
grid_rf.fit(X_train, y_train)

# Kết quả tốt nhất
print("Best Parameters:", grid_rf.best_params_)
print("Best CV Score:", grid_rf.best_score_)
best_rf = grid_rf.best_estimator_

y_pred_rf = best_rf.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))
import pandas as pd

results_rf = pd.DataFrame(grid_rf.cv_results_)
print(results_rf[['param_n_estimators', 'param_max_depth', 'mean_test_score']])