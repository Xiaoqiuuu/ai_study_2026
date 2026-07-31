import pandas as pd
import joblib
from utils import extract_features

# 加载模型
model = joblib.load('best_model_v2.pkl')

# 加载测试集
test_df = pd.read_csv("../data/test.csv")
X_test = extract_features(test_df)

# 预测
predictions = model.predict(X_test)

# 生成提交文件
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': predictions
})
submission.to_csv('submission.csv', index=False)
print("提交文件已生成！前5行：")
print(submission.head())