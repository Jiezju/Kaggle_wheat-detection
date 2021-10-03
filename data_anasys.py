import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from PIL import Image

DIR_INPUT = './data'
DIR_TRAIN = f'{DIR_INPUT}/train'
DIR_TEST = f'{DIR_INPUT}/test'

# pandas 读取数据
'''
查看数据组成
'''
train_df = pd.read_csv('./data/train.csv')
print(train_df.head())

print('=' * 50)
'''
bbox 属性 转为 [x,y,w,h]
'''
train_df[['x','y','w','h']] = 0
train_df[['x','y','w','h']] = np.stack(train_df['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=','))).astype(np.float32)
train_df.drop(columns=['bbox'], inplace=True)
print(train_df.head())

'''
数据量
'''
print(train_df.shape)

print('=' * 50)

'''
查看未标注数据量 说明 有 3422-3373=49 张图片没有标注 
'''
unlebel = len(os.listdir(DIR_TRAIN)) - train_df['image_id'].nunique()
print(unlebel)

'''
查看标注的框的分布范围  一张图最多的有 116 个标注
'''
# 查看标注数量的分布情况
counts = train_df['image_id'].value_counts()
print(f'number of boxes, range [{min(counts)}, {max(counts)}]')
sns.displot(counts, kde=False)
plt.xlabel('boxes')
plt.ylabel('images')
plt.title('boxes vs. images')
plt.show()

# 查看标注坐标和宽高的分布情况
train_df['cx'] = train_df['x'] + train_df['w'] / 2
train_df['cy'] = train_df['y'] + train_df['h'] / 2

ax = plt.subplots(1, 4, figsize=(16, 4), tight_layout=True)[1].ravel()
ax[0].set_title('x vs. y')
ax[0].set_xlim(0, 1024)
ax[0].set_ylim(0, 1024)
ax[1].set_title('cx vs. cy')
ax[1].set_xlim(0, 1024)
ax[1].set_ylim(0, 1024)
ax[2].set_title('w vs. h')
ax[3].set_title('area size')
sns.histplot(data=train_df, x='x', y='y', ax=ax[0], bins=50, pmax=0.9)
sns.histplot(data=train_df, x='cx', y='cy', ax=ax[1], bins=50, pmax=0.9)
sns.histplot(data=train_df, x='w', y='h', ax=ax[2], bins=50, pmax=0.9)
sns.histplot(train_df['w'] * train_df['h'], ax=ax[3], bins=50, kde=False)
plt.show()

