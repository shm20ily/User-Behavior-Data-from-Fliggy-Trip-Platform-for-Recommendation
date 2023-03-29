import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

data = pd.read_csv("behavior_pred.csv")
pd.set_option('display.max_columns', None)
data.drop(['Unnamed: 0', 'timestamp', 'date', 'time'], axis=1, inplace=True)
# print(data.info())

X = data.iloc[:, [0, 1, 3, 4, 5, 6, 7]].values
Y = data.iloc[:, 2].values
# 将数据打乱
X, Y = shuffle(X, Y, random_state=13)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
print('预测值:', y_pred)

# 绘制混淆矩阵的热力图
f, ax = plt.subplots()
c = confusion_matrix(y_test, y_pred, labels=[1, 2, 3, 4])
print(c)
# 画热力图
sns.heatmap(c, annot=True, ax=ax)
# 标题
ax.set_title('confusion matrix')
# x轴
ax.set_xlabel('predict')
# y轴
ax.set_ylabel('true')
plt.savefig('./images/混淆矩阵.png')
plt.show()

# 1.准确率
print('准确率:', accuracy_score(y_test, y_pred))

# 2.分类总指标
print('分类总指标:', classification_report(y_test, y_pred))
