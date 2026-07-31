import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

#============加载数据=============
train_df = pd.read_csv("../data/train.csv")
print(f"数据形状: {train_df.shape}")
print(f"存活率: {train_df['Survived'].mean():.2f}")

#============选择特征==============
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
x = train_df[features]
y = train_df['Survived']

#===========定义预处理=============
#数值特征: 填充缺失值(中位数) -> 标准化
numeric_features = ['Age', 'SibSp', 'Parch', 'Fare']
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy = 'median')),
    ('scaler', StandardScaler())
])

#类别特征: 填充缺失数(众数) -> OneHot编码
categorical_features = ['Pclass', 'Sex', 'Embarked']
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy = 'most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown = 'ignore', sparse_output = False)) 
])

#组合
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

#==================模型对比================
models = {
    'LogisticRegression': LogisticRegression(max_iter= 1000, random_state= 42),
    'RandomForest': RandomForestClassifier(n_estimators = 100, random_state = 42)
}

print("\n=== 模型对比(5折交叉验证)===")
for name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    scores = cross_val_score(pipeline, x, y, cv = 5, scoring = 'accuracy')
    print(f"{name:20s}: {scores.mean():.4f} (+/- {scores.std():.4f})")

#================GridSearch调参====================
print("\n=== RandomForest 调参 ===")
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [5, 10, None],
    'classifier__min_samples_split': [2, 5]
}

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state= 42))
])

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv= 5,
    scoring= 'accuracy',
    n_jobs=-1,
    verbose = 1
)
grid.fit(x,y)


print(f"最优参数: {grid.best_params_}")
print(f"最优 CV 分数: {grid.best_score_:.4f}")

# ========== 保存最佳模型 ==========
import joblib
joblib.dump(grid.best_estimator_, 'best_model.pkl')
print("\n模型已保存")