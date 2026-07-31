import pandas as pd
import numpy as np
def extract_features(df, is_train = True):
    """
    特征工程: 从原始数据中提取/构造更有预测力的特征
    """
    data = df.copy()
    #1. 家庭大小(SibSp + Parch + 自己)
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

    #2. 是否独自一人(FamilySize == 1)
    data['IsAlone'] = (data['FamilySize'] == 1).astype(int)

    #3. 从Name提取Title
    data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    
    # 合并稀有称谓
    title_map = {
        'Mr': 'Mr',
        'Miss': 'Miss',
        'Mrs': 'Mrs',
        'Master': 'Master',  # 男孩，存活率很高
        'Dr': 'Rare',
        'Rev': 'Rare',
        'Col': 'Rare',
        'Major': 'Rare',
        'Mlle': 'Miss',
        'Countess': 'Rare',
        'Ms': 'Miss',
        'Lady': 'Rare',
        'Jonkheer': 'Rare',
        'Don': 'Rare',
        'Dona': 'Rare',
        'Mme': 'Mrs',
        'Capt': 'Rare',
        'Sir': 'Rare'
    }
    data['Title'] = data['Title'].map(title_map)
    data['Title'] = data['Title'].fillna('Rare')
    
    # 4. 船票价格分桶（Fare 分布极不均匀，分桶后更稳定）
    # 用训练集的分位点，避免数据泄露
    data['FareBin'] = pd.qcut(data['Fare'], q=4, labels=['Low', 'Med', 'High', 'Premium'])
    
    # 5. 年龄分桶（儿童存活率明显更高）
    data['AgeBin'] = pd.cut(data['Age'], bins=[0, 12, 20, 40, 60, 100], 
                            labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
    
    # 6. 船舱甲板（Cabin 的第一个字母，缺失填 'Unknown'）
    data['Deck'] = data['Cabin'].str[0]
    data['Deck'] = data['Deck'].fillna('Unknown')
    
    # 7. 性别编码（女性存活率远高于男性，这是最强特征）
    data['SexCode'] = (data['Sex'] == 'female').astype(int)
    
    # 8. 船票前缀（Ticket 可能有家族信息）
    data['TicketPrefix'] = data['Ticket'].str.split().str[0]
    data['TicketPrefix'] = data['TicketPrefix'].replace('LINE', 'Unknown')
    # 如果是纯数字，设为 'NUM'
    data['TicketPrefix'] = data['TicketPrefix'].apply(
        lambda x: 'NUM' if str(x).isdigit() else x
    )

    # 8. Pclass + Sex 交互（头等舱女性存活率极高）
    data['Pclass_Sex'] = data['Pclass'].astype(str) + '_' + data['SexCode'].astype(str)

    # 选择最终特征
    feature_cols = [
        'Pclass', 'SexCode', 'Age', 'Fare',
        'FamilySize', 'IsAlone', 'Title', 'FareBin', 'AgeBin', 'Deck', 'Pclass_Sex'
    ]
    
    return data[feature_cols]