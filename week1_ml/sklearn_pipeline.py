import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


#=================加载数据================
iris = load_iris()
x, y = iris.data, iris.target

#=================划分数据================
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, random_state = 42, stratify = y  #stratify保证类别一致
)

#==============构建pipeline===============
#Pipeline = 预处理 + 模型, 串在一起防止数据泄露
Pipeline = Pipeline([
    ('scaler', StandardScaler()),  #步骤1：标准化
    # ('poly', PolynomialFeatures(degree = 2, include_bias = False)),  #步骤2：多项式特征
    ('classifier', LogisticRegression(max_iter = 1000, random_state = 42))  #步骤3：模型
])

#==============交叉验证====================
#5折交叉验证: 把训练集分成5份，轮流用4份训练，1份验证
cv_scores = cross_val_score(Pipeline, x_train, y_train, cv = 5, scoring = 'accuracy')
print(f"5折交叉验证准确率: {cv_scores.mean(): .4f} (+/- {cv_scores.std(): .4f})")

#=============网格搜索调参=================
param_grid = {
    # 'poly__degree': [1, 2],
    'classifier__C': [0.01, 0.1, 1.0, 10.0],
    'classifier__penalty': [ 'l2']
}

grid_search = GridSearchCV(
    Pipeline,
    param_grid,
    cv = 5,
    scoring = 'accuracy',
    n_jobs=-1,
    verbose = 1
)

grid_search.fit(x_train, y_train)

print(f"\n最优参数: {grid_search.best_params_}")
print(f"最优交叉验证得分: {grid_search.best_score_:.4f}")

#=============在测试集上评估==============
best_model = grid_search.best_estimator_
y_pred = best_model.predict(x_test)

print(f"\n测试集准确率: {accuracy_score(y_test, y_pred):.4f}")
print(f"分类报告:\n{classification_report(y_test, y_pred, target_names=iris.target_names)}")

#混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print(f"混淆矩阵:\n{cm}")

#==============保存模型================
import joblib
joblib.dump(best_model, 'best_model.pkl')