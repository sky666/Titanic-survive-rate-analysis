'''
分析步骤：
1.提出问题（Business Understanding ）
2.理解数据（Data Understanding）
    采集数据
    导入数据
    查看数据集信息
3.数据清洗（Data Preparation ）
    数据预处理
    特征工程（Feature Engineering）
4.构建模型（Modeling）
5.模型评估（Evaluation）
6.方案实施 （Deployment）
    提交结果到Kaggle
    报告撰写
'''
# 导入处理数据包
import numpy as np
import pandas as pd

# 1 数据理解
# 导入数据
train = pd.read_csv('./train.csv')  # 训练数据集
test = pd.read_csv('./test.csv')  # 测试数据集
print('训练数据集:', train.shape, '测试数据集:', test.shape)
print('训练数据集有多少行数据:', train.shape[0], '测试数据集有多少行数据:', test.shape[0])

# 合并数据
full = pd.concat([train, test], ignore_index=True)  # 合并数据方式 1
# full = train.append(test, ignore_index=True)  # 合并数据方式 2
print('合并数据集:', full.shape)
# print(full.head())

# 查看数据集信息
print(full.describe())  # 获取数据类型列的描述统计信息
print(full.info())  # 查看每一列的数据类型，和数据总数
'''
数据类型：
Age 里面数据总数是1046条，缺失了1309-1046=263，缺失率263/1309=20%
Fare 里面数据总数是1308条，缺失了1条数据

字符串类型：
Embarked 里面数据总数是1307，缺失了2条数据
Cabin 里面数据总数是295，缺失了1309-295=1014，缺失率=1014/1309=77.5%
'''


# 2 数据准备
# 数据清洗
# 数据预处理---缺失值处理
# 对于数据类型，处理缺失值最简单的方法就是用平均数来填充缺失值
full['Age'] = full['Age'].fillna(full['Age'].mean())  # 对 Age 缺失值用平均数填充
full['Fare'] = full['Fare'].fillna(full['Fare'].mean())  # 对 Fare 缺失值用平均数填充
print('数据类型处理后:')
print(full.info())

# 对于字符串类型，使用最频繁出现的值来填充缺失值
print(full['Embarked'].value_counts())  # 查看Embarked 中不同元素的个数，S为最多的元素
full['Embarked'] = full['Embarked'].fillna('S')  # 将缺失值填充为最频繁出现的值
full['Cabin'] = full['Cabin'].fillna('U')  # 缺失数据比较多，船舱号（Cabin）缺失值填充为U，表示未知（Uknow）
print('字符串类型处理后:')
print(full.info())

# 特征提取
# 规则：数值类型直接使用; 时间序列转成年月日; 分类数据用数值代替类别
# 分类数据之有直接类别的： 乘客性别(Sex), 登船港口(Embarked), 客舱等级(Pclass)
# 将Sex的值映射为数值  male对应数值1，female对应数值0
sex_mapdict = {'male': 1, 'female': 0}
full['Sex'] = full['Sex'].map(sex_mapdict)  # map函数：对Series每个数据应用自定义的字典进行计算

# 对Embarked进行one-hot编码，产生虚拟变量
embarked_df = pd.get_dummies(full['Embarked'], prefix='Embarked')  # get_dummies函数：进行one-hot编码，产生虚拟变量（dummy variables）
full = pd.concat([full, embarked_df], axis=1)  # 添加one-hot编码产生的虚拟变量（dummy variables）到泰坦尼克号数据集full
full.drop('Embarked', axis=1, inplace=True)  # 把登船港口(Embarked)删掉

# 对Pclass进行one-hot编码，产生虚拟变量
pclass_df = pd.get_dummies(full['Pclass'], prefix='Pclass')  # get_dummies函数：进行one-hot编码，产生虚拟变量（dummy variables）
full = pd.concat([full, pclass_df], axis=1)  # 添加one-hot编码产生的虚拟变量（dummy variables）到泰坦尼克号数据集full
full.drop('Pclass', axis=1, inplace=True)  # 把客舱等级（Pclass）删掉

# 分类数据之子符串类型： 乘客姓名(Name), 客舱号(Cabin), 船票编号(Ticket)
# 从姓名中提取头衔
# 定义函数：从姓名中获取头衔
def get_title(name):
    str1 = name.split(',')[1]  # 获取name后面的字符，如：Mr. Owen Harris
    str2 = str1.split('.')[0]  # 获取str1 前面的字符, 如：Mr
    str3 = str2.strip()  # 去除str2字符中前后的空格符
    return str3

title_df = full['Name'].apply(get_title)  # apply函数：对Series每个数据使用定义的函数进行计算

# 姓名中头衔字符串与定义头衔类别的映射关系
title_mapDict = {
                    "Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty"
                    }

title_df = title_df.map(title_mapDict)  # map函数：对Series每个数据应用自定义的字典进行计算
title_df = pd.get_dummies(title_df)  # 使用get_dummies进行one-hot编码,产生虚拟变量（dummy variables）
full = pd.concat([full, title_df], axis=1)  # 添加one-hot编码产生的虚拟变量（dummy variables）到泰坦尼克号数据集full
full.drop('Name', axis=1, inplace=True)  # 删掉姓名这一列

# 从客舱号中提取客舱类别
full['Cabin'] = full['Cabin'].apply(lambda a: a[0])  # 提取客舱的首字母
cabin_df = pd.get_dummies(full['Cabin'], prefix='Cabin')  # 使用get_dummies进行one-hot编码,产生虚拟变量（dummy variables）
full = pd.concat([full, cabin_df], axis=1)  # 添加one-hot编码产生的虚拟变量（dummy variables）到泰坦尼克号数据集full
full.drop('Cabin', axis=1, inplace=True)

# 建立家庭人数和家庭类别
# 家庭人数=同代直系亲属数（Parch）+不同代直系亲属数（SibSp）+乘客自己
'''
家庭类别：
小家庭Family_Single：家庭人数=1
中等家庭Family_Small: 2<=家庭人数<=4
大家庭Family_Large: 家庭人数>=5
'''

family_df = pd.DataFrame()
family_df['Family_size'] = full['SibSp'] + full['Parch'] + 1  # 家庭人数
family_df['Family_Single'] = family_df['Family_size'].apply(lambda s: 1 if s ==1 else 0)  # 小家庭
family_df['Family_Small'] = family_df['Family_size'].apply(lambda s: 1 if (s >= 2 and s <= 4) else 0)  # 中等家庭
family_df['Family_Large'] = family_df['Family_size'].apply(lambda s: 1 if s >= 5 else 0)  # 大家庭

full = pd.concat([full, family_df], axis=1)  # 添加 family_df 到泰坦尼克号数据集 full

# 特征选择
corr_df = full.corr()  # 相关性矩阵
corr_df.sort_values(by='Survived', ascending=False, inplace=True)  # 查看各个特征与生还情况（Survived）的相关系数,按降序排列
'''
根据各个特征与生成情况（Survived）的相关系数大小，选择一下几个特征作为模型的输入：
头衔、客舱等级、家庭大小、船票价格、船舱号、登船港口、性别
'''

full_choice = pd.concat([title_df,  # 头衔
                         pclass_df,  # 客舱等级
                         family_df,  # 家庭大小
                         full['Fare'],  # 船票价格
                         cabin_df,  # 船舱号
                         embarked_df,  # 登船港口
                         full['Sex'],  # 性别
                         ], axis=1)
# print(full_choice)  # 特征选择生成的新数据


# 3 模型建立
# 建立训练数据集和测试数据集
# 原始数据集：特征
source_x = full_choice.loc[0: train.shape[0]-1, :]
# 原始数据集：标签
source_y = full.loc[0: train.shape[0]-1, 'Survived']
# 预测数据集：特征
pred_x = full_choice.loc[train.shape[0]: , :]

'''
从原始数据集（source）中拆分出训练数据集（用于模型训练train），测试数据集（用于模型评估test）
train_test_split是交叉验证中常用的函数，功能是从样本中随机的按比例选取train data和test data
train_data：所要划分的样本特征集
train_target：所要划分的样本结果
test_size：样本占比，如果是整数的话就是样本的数量
'''
from sklearn.model_selection import train_test_split
# 建立模型用的训练数据集和测试数据集
train_x, test_x, train_y, test_y = train_test_split(source_x, source_y, train_size=0.8)
print('原始数据集特征：', source_x.shape,
      '训练数据集特征：', train_x.shape,
      '测试数据集特征：', test_x.shape,)

print('原始数据集标签：', source_y.shape,
      '训练数据集标签：', train_y.shape,
      '测试数据集标签：', test_y.shape,)

# 选择机器学习算法
from sklearn.linear_model import LogisticRegression  # 导入逻辑回归算法
model = LogisticRegression()  # 创建模型：逻辑回归
# from sklearn.ensemble import RandomForestClassifier  # 导入随机森林算法
# model = RandomForestClassifier(n_estimators=100)  # 创建模型：随机森林
# from sklearn.svm import SVC, LinearSVC  # 导入支持向量机算法
# model = SVC()  # 创建模型：支持向量机
# from sklearn.ensemble import GradientBoostingClassifier  # 导入Gradient Boosting Classifier算法
# model = GradientBoostingClassifier()  # 创建模型：Gradient Boosting Classifier
# from sklearn.neighbors import KNeighborsClassifier  # 导入K-nearest neighbors算法
# model = KNeighborsClassifier(n_neighbors = 3)  # 创建模型：K-nearest neighbors
# from sklearn.naive_bayes import GaussianNB  # 导入Gaussian Naive Bayes算法
# model = GaussianNB()  # 创建模型：Gaussian Naive Bayes

# 训练模型
model.fit(train_x, train_y)


# 4 模型评估
# 评估模型
score = model.score(test_x, test_y)
print(score)


# 5 方案实施
# 方案实施
pred_y = model.predict(pred_x)  # 生成的预测值是浮点数（0.0,1,0）,需要转换为整型
pred_y = pred_y.astype(int)
passenger_id = full.loc[train.shape[0]:, 'PassengerId']  # 提取test中的PassengerId
pred_df = pd.DataFrame({'PassengerId': passenger_id, 'Survived': pred_y})  #预测数据
# print(pred_df)
# 保存预测数据
pred_df.to_csv('pred_test.csv', encoding='utf_8_sig', index=False)