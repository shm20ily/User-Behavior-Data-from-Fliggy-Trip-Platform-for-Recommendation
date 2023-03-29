import pandas as pd
import datetime
import matplotlib.pyplot as plt

# 加载字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
# 显示负号
plt.rcParams['axes.unicode_minus'] = False

# 增加列名'user_id', 'item_id', 'behavior_type','timestamp'
reader = pd.read_csv('user_item_behavior_history.csv', header=None,
                     names=['user_id', 'item_id', 'behavior_type', 'timestamp'], iterator=True)

# 使用get_chunk方法获取数据
loop = True
chunkSize = 10000000  # 设置chunksize
chunks = []

# start time
starttime = datetime.datetime.now()

# long running
while loop:
    try:
        chunk = reader.get_chunk(chunkSize)
        chunks.append(chunk)
    except StopIteration:
        loop = False
        print("Iteration is stopped.")  # 迭代完成
# 拼接chunks
df = pd.concat(chunks, ignore_index=True)

endtime = datetime.datetime.now()  # end time

# 共计数据获取时间
# print('loop_time:', (endtime - starttime).seconds)

# timestamp转为datetime
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
# 查看前五行数据
# print(df.head())

start_date = pd.to_datetime('20210503', format='%Y-%m-%d %H:%M:%S')  # 开始日期
end_date = pd.to_datetime('20210604', format='%Y-%m-%d %H:%M:%S')  # 截止时间
df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
df = df.sort_values(by='timestamp', ascending=True)  # 时间按升序排列
df = df.reset_index(drop=True)  # 重置索引
# print(df.head())

# 把列'nehavior_type'中的'clk','fav','cart','pay'替换为1,2,3,4
'''
1——点击
2——收藏
3——加购
4——购买
'''
replace_values = {'clk': 1, 'fav': 2, 'cart': 3, 'pay': 4}
df['behavior_type'] = df['behavior_type'].replace(replace_values)

# 时间格式转换，获取日期、时间、年、月、日、周几、小时
df['date'] = df['timestamp'].dt.date
df['time'] = df['timestamp'].dt.time
df['year'] = df['timestamp'].dt.year
df['month'] = df['timestamp'].dt.month
df['day'] = df['timestamp'].dt.day
df['weekday'] = df['timestamp'].dt.strftime("%w")
df['hour'] = df['timestamp'].dt.hour
pd.set_option('display.max_columns', None)
# print(df.head())

'''从行为类型中各取一万数据条，以供模型训练与测试'''
data1 = df[df['behavior_type'] == 1].head(10000)
data2 = df[df['behavior_type'] == 2].head(10000)
data3 = df[df['behavior_type'] == 3].head(10000)
data4 = df[df['behavior_type'] == 4].head(10000)
data = pd.concat([data1, data2, data3, data4], axis=0)
data.to_csv('behavior_pred.csv')
print('data_save_OK')

# '''用户维度----------浏览量pv、访客量uv、成交量分析'''
# '''总量'''
uv = df['user_id'].nunique()  # 用户数量uv
item_num = df['item_id'].nunique()  # 商品数量
behavior_num = df['behavior_type'].count()  # 行为总数
# print('用户数量：', uv)
# print('商品总数：', item_num)
# print('行为总数：', behavior_num)

clk_data = df[df['behavior_type'] == 1]  # 点击数据
fav_data = df[df['behavior_type'] == 2]  # 收藏数据
cart_data = df[df['behavior_type'] == 3]  # 加购数据
pay_data = df[df['behavior_type'] == 4]  # 支付数据
# print(len(clk_data), len(fav_data), len(cart_data), len(pay_data))

'''点击clk相当于浏览pv'''
page_view = len(clk_data)  # 总浏览量
pay_num = len(pay_data)  # 总支付量
pv_avg = round(page_view / uv, 2)  # 总平均浏览量
pay_avg = round(pay_num / uv, 2)  # 总平均支付量
# print('总浏览量:{}\n总支付量:{}\n总平均浏览量:{}\n总平均支付量:{}'.format(page_view, pay_num, pv_avg, pay_avg))

'''日均'''
# 日访问量
pv_daily = clk_data.groupby('date')['user_id'].count().reset_index().rename(columns={'user_id': 'pv_daily'})
# 日访客量
uv_daily = df.groupby('date')['user_id'].apply(lambda x: x.drop_duplicates().count()).reset_index().rename(
    columns={'user_id': 'uv_daily'})
# 拼接
daily_data = pd.merge(pv_daily, uv_daily, how='outer', on='date')
# 日平均访问量
daily_data['avg_pv_daily'] = round(daily_data['pv_daily'] / daily_data['uv_daily'], 2)
# 日成交量
pay_daily = pay_data.groupby('date')['user_id'].count().reset_index().rename(columns={'user_id': 'pay_daily'})
# 拼接
daily_data = pd.merge(daily_data, pay_daily, on='date')
# 日平均成交量
daily_data['avg_pay_daily'] = round(daily_data['pay_daily'] / daily_data['uv_daily'], 2)
# print(daily_data.head())

# 图1
# 创建图形
fig = plt.figure(figsize=(20, 8), dpi=80)
plt.grid(True, linestyle="--", alpha=0.5)
ax1 = fig.add_subplot(111)
ax1.plot(daily_data.date, daily_data.avg_pv_daily, label='日平均访问量')
ax1.set_ylabel('日平均访问量')
ax1.set_title('daily_data', size=15)
ax1.legend(loc='upper left')
ax2 = ax1.twinx()  # 共享x轴
ax2.plot(daily_data.date, daily_data.avg_pay_daily, 'r', label='日平均成交量')
ax2.set_ylabel('日平均成交量')
ax2.set_xlabel('日期', size=15)
ax2.legend(loc='upper right')
plt.savefig('./images/日平均访问量与成交量.png')
plt.show()
print('图1', '-' * 80)

# 每小时访问量
pv_hour = clk_data.groupby('hour')['user_id'].count().reset_index().rename(columns={'user_id': 'pv_hour'})
# 每小时访客量
uv_hour = df.groupby('hour')['user_id'].apply(lambda x: x.drop_duplicates().count()).reset_index().rename(
    columns={'user_id': 'uv_hour'})
# 拼接
hour_data = pd.merge(pv_hour, uv_hour, how='outer', on='hour')
# 每小时平均访问量
hour_data['avg_hour_pv'] = round(hour_data['pv_hour'] / hour_data['uv_hour'], 2)
# print(hour_data.head())

# 每小时成交量
pay_hour = pay_data.groupby('hour')['user_id'].count().reset_index().rename(columns={'user_id': 'pay_hour'})
# 拼接
hour_data = pd.merge(hour_data, pay_hour, on='hour')
# 每小时平均成交量
hour_data['avg_hour_pay'] = round(hour_data['pay_hour'] / hour_data['uv_hour'], 2)

# 图2
# 创建图形
fig = plt.figure(figsize=(20, 8), dpi=80)
plt.grid(True, linestyle="--", alpha=0.5)
ax1 = fig.add_subplot(111)
ax1.plot(hour_data.hour, hour_data.avg_hour_pv, label='每小时平均访问量')
ax1.set_ylabel('每小时平均访问量')
ax1.set_title('hour_data', size=15)
ax1.legend(loc='upper left')
ax2 = ax1.twinx()  # 共享x轴
ax2.plot(hour_data.hour, hour_data.avg_hour_pay, 'r', label='每小时平均成交量')
ax2.set_ylabel('每小时平均成交量')
ax2.set_xlabel('时刻', size=15)
ax2.legend(loc='upper right')
plt.savefig('./images/每小时平均访问量与成交量.png')
plt.show()
print('图2', '-' * 80)

'''周均'''
pv_weekday = clk_data.groupby('weekday')['user_id'].count().reset_index().rename(columns={'user_id': 'pv_weekday'})
uv_weekday = df.groupby('weekday')['user_id'].apply(lambda x: x.drop_duplicates().count()).reset_index().rename(
    columns={'user_id': 'uv_weekday'})
weekday_data = pd.merge(pv_weekday, uv_weekday, how='outer', on='weekday')
weekday_data['avg_pv_weekday'] = round(weekday_data['pv_weekday'] / weekday_data['uv_weekday'], 2)
# print(weekday_data.head())

pay_weekday = pay_data.groupby('weekday')['user_id'].count().reset_index().rename(columns={'user_id': 'pay_weekday'})
weekday_data = pd.merge(weekday_data, pay_weekday, on='weekday')
weekday_data['avg_pay_weekday'] = round(weekday_data['pay_weekday'] / weekday_data['uv_weekday'], 2)

# 图3
# 创建图形
fig = plt.figure(figsize=(20, 8), dpi=80)
plt.grid(True, linestyle="--", alpha=0.5)
ax1 = fig.add_subplot(111)
ax1.plot(weekday_data.weekday, weekday_data.avg_pv_weekday, label='每周几平均访问量')
ax1.set_ylabel('每周几平均访问量')
ax1.set_title('weekday_data', size=15)
ax1.legend(loc='upper left')
ax2 = ax1.twinx()  # 共享x轴
ax2.plot(weekday_data.weekday, weekday_data.avg_pay_weekday, 'r', label='每周几平均成交量')
ax2.set_ylabel('每周几平均成交量')
ax2.set_xlabel('周几', size=15)
ax2.legend(loc='upper right')
plt.savefig('./images/每周几平均访问量与成交量.png')
plt.show()
print('图3', '-' * 80)

'''用户画像----只对支付用户进行分析'''
# 支付用户的消费时间
payuser_date = pay_data[['user_id', 'hour', 'weekday']]
# 支付用户按小时分布
payuser_hour = payuser_date.groupby('hour')['user_id'].apply(
    lambda x: x.drop_duplicates().count()).reset_index().rename(columns={'user_id': 'num'})

# 图4
# 创建图形
plt.figure(figsize=(20, 8), dpi=80)
plt.bar(payuser_hour.hour, payuser_hour.num, label='每小时成交量')
plt.plot(payuser_hour.hour, payuser_hour.num, 'ro-', color='r', alpha=0.8, linewidth=3)
# 添加标注
plt.xlabel('小时', size=15)
plt.ylabel('成交量', size=15)
# 添加标题
plt.title('支付用户按小时分布', size=15)
# 添加图例
plt.legend(loc='best')
plt.savefig('./images/支付用户按小时分布.png')
# 显示柱状-折线图
plt.show()
print('图4', '-' * 80)

# 支付用户按周几分布
payuser_weekday = payuser_date.groupby('weekday')['user_id'].apply(
    lambda x: x.drop_duplicates().count()).reset_index().rename(columns={'user_id': 'num'})
# print(payuser_weekday)

# 图5
# 创建图形
plt.figure(figsize=(10, 5), dpi=80)
plt.bar(payuser_weekday.weekday, payuser_weekday.num, label='每周几成交量')
# 添加标注
plt.xlabel('星期', size=15)
plt.ylabel('成交量', size=15)
# 添加标题
plt.title('支付用户按星期分布', size=15)
# 添加图例
plt.legend(loc='best')
plt.savefig('./images/支付用户按星期分布.png')
# 显示柱状图
plt.show()
print('图5', '-' * 80)

# 读取user_profile.csv，并增加列名'user_id','age','sex','career','use_city_id','crowd_label'
user_profile = pd.read_csv('user_profile.csv', header=None,
                           names=['user_id', 'age', 'sex', 'career', 'use_city_id', 'crowd_label'], encoding='utf-8')
# 根据user_id拼接user_profile数据
payuser_profile = pd.merge(payuser_date, user_profile, how='left', on='user_id')

# 最大年龄及最小年龄
# print(payuser_profile['age'].max(), payuser_profile['age'].min())
''' 考虑到平台注册年龄限制，所以最小年龄设置为18岁; 对年龄异常数据进行处理'''
payuser_profile = payuser_profile[payuser_profile['age'] >= 18]
# print(payuser_profile['age'].min())

# 年龄分箱
bins = [18, 25, 35, 45, 60, 80, 125]
labels = ['[18-25)', '[25-35)', '[35-45)', '[45-60)', '[60-80)', '[80-125]']
payuser_profile['age_cut'] = pd.cut(x=payuser_profile.age, bins=bins, right=False, labels=labels)
# print(payuser_profile.head())
# 年龄分布
payuser_age = payuser_profile[['user_id', 'age_cut']].groupby('age_cut')['user_id'].apply(
    lambda x: x.drop_duplicates().count()).reset_index().rename(columns={'user_id': 'num'})
# print(payuser_age)

# 图6
# 创建图形
plt.figure(figsize=(10, 5), dpi=80)
plt.bar(payuser_age.age_cut, payuser_age.num, label='年龄段成交量')
# 添加标注
plt.xlabel('年龄', size=15)
plt.ylabel('成交量', size=15)
# 添加标题
plt.title('支付用户按年龄段分布', size=15)
# 添加图例
plt.legend(loc='best')
plt.savefig('./images/支付用户按年龄段分布.png')
# 显示柱状图
plt.show()
print('图6', '-' * 80)

# 性别分布
payuser_sex = payuser_profile[['user_id', 'sex']].groupby('sex')['user_id'].apply(
    lambda x: x.drop_duplicates().count()).reset_index().rename(columns={'user_id': 'num'})
# print(payuser_sex)

# 图7
# 画饼图
plt.pie(payuser_sex['num'], labels=payuser_sex['sex'], autopct='%1.2f%%')
plt.title('支付用户按性别分布')
plt.legend(loc='best')
plt.savefig('./images/支付用户按性别分布.png')
plt.show()
print('图7', '-' * 80)
