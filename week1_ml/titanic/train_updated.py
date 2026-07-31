import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# 加载数据
train_df = pd.read_csv("../data/train.csv")
print(f"数据形状: {train_df.shape}")
print(f"存活率: {train_df['Survived'].mean():.2f}")

# 特征工程
from utils import extract_features
X = extract_features(train_df)
y = train_df['Survived']

# 区分特征类型
numeric_features = ['Age', 'Fare', 'FamilySize']
ordinal_features = ['Pclass', 'SexCode', 'IsAlone']  # 有大小关系
categorical_features = ['Title', 'FareBin', 'AgeBin', 'Deck']  # 无大小关系

# 预处理流程
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

ordinal_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('ord', ordinal_transformer, ordinal_features),
    ('cat', categorical_transformer, categorical_features)
])

# ========== 模型对比 ==========
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42),
    'XGBoost': XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
}

print("\n=== 模型对比（5折交叉验证）===")
for name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy', n_jobs=-1)
    print(f"{name:20s}: {scores.mean():.4f} (+/- {scores.std():.4f})")

# ========== XGBoost 精细调参 ==========
print("\n=== XGBoost 精细调参 ===")
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [3, 5, 7],
    'classifier__learning_rate': [0.01, 0.1, 0.2],
    'classifier__subsample': [0.8, 1.0],  # 防止过拟合
    'classifier__colsample_bytree': [0.8, 1.0]
}

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'))
])

grid = GridSearchCV(
    pipeline, param_grid, 
    cv=5, scoring='accuracy', 
    n_jobs=-1, verbose=1
)
grid.fit(X, y)

print(f"\n最优参数: {grid.best_params_}")
print(f"最优 CV 分数: {grid.best_score_:.4f}")

# 保存最佳模型
joblib.dump(grid.best_estimator_, 'best_model_v2.pkl')
print("\n模型已保存: best_model_v2.pkl")