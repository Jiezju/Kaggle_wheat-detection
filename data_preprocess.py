# 数据集划分 与 增广

# 导入依赖的库
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import random
import cv2
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import os
import time
import datetime
import glob
import warnings

warnings.filterwarnings("ignore")

# 导入数据
data = pd.read_csv(r'./data/train.csv')
print(data.head(2))

# 注意 bbox 是一个 string 类型，所以需要将它转换为 ndarray
bboxs = np.stack(data['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))
print(bboxs)

# 将 bbox 转换为 x,y,w,h 四列，然后删除 bbox 列
for i, column in enumerate(['x', 'y', 'w', 'h']):
    data[column] = bboxs[:, i]
data.drop(columns=['bbox'], inplace=True)
print(data.head(2))

'''
数据集划分

# 这么做的目的来源于两个方面。
# 1. 需要保证划分的多折训练集中数据来源占比一致。
# 2. 需要保证划分的多折训练集中 bbox 分布大致一致。 
'''
# 利用 sklearn 生成 5 折的分层交叉验证的实例。
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 复制 image_id
df_folds = data[['image_id']].copy()

# 设定每一个 image_id 的 bbox个数（这个时候image_id 是有重复的）
df_folds.loc[:, 'bbox_count'] = 1

# 按照 image_id 聚合，得到每一个 image_id 的 bbox 的个数
df_folds = df_folds.groupby('image_id').count()

# 取 source
df_folds.loc[:, 'source'] = data[['image_id', 'source']].groupby('image_id').min()['source']

# 按照 source 和 bbox_count 的个数划分为了 34 个 group
df_folds.loc[:, 'stratify_group'] = np.char.add(
    df_folds['source'].values.astype(str),
    df_folds['bbox_count'].apply(lambda x: f'_{x // 15}').values.astype(str)
)

print(np.unique(df_folds.stratify_group))

# 设定默认的 fold 为 0
df_folds.loc[:, 'fold'] = 0

# 进行分层的交叉验证，将 ‘stratify_group’ 做为 y。这样就保证了划分的一致性。
for fold_number, (train_index, val_index) in enumerate(skf.split(X=df_folds.index, y=df_folds['stratify_group'])):
    df_folds.loc[df_folds.iloc[val_index].index, 'fold'] = fold_number

print(df_folds.head(5))

'''
albumentations数据增强 (https://albumentations.readthedocs.io/en/latest/)

- Compose 函数为要对图像实行的变换。 Compose 包含图像的变换和 Bbox 的变换。
- p 为实现该变换的概率。
- OneOf 表示只选择一个变换来实现，这个时候概率需要归一化。
- min_area 表示 Bbox 所占像素小于这个值的会被抛弃掉。
- min_visibility 表示 Bbox 占图片比例小于这个值的会被抛弃掉。
'''


# 训练集阶段的数据增强变换
# 依托于 albumentations 这个三方包
def get_train_transforms():
    return A.Compose(
        [
            A.RandomSizedCrop(min_max_height=(800, 800), height=1024, width=1024, p=0.5),
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=0.1, sat_shift_limit=0.3,
                                     val_shift_limit=0.3, p=0.9),
                A.RandomBrightnessContrast(brightness_limit=0.4,
                                           contrast_limit=0.3, p=0.9),
            ], p=0.9),
            A.ToGray(p=0.01),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Resize(height=512, width=512, p=1),
            A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )
    )


# 验证集阶段的数据增强变换
# 依托于 albumentations 这个三方包
def get_valid_transforms():
    return A.Compose(
        [
            A.Resize(height=512, width=512, p=1.0),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )
    )

'''
dataset 生成器
'''


# Torch 的数据生成器
class WheatData(Dataset):

    def __init__(self, data, image_ids, transforms=None, test=False):
        super().__init__()

        self.image_root = './data/train'
        # 图片的 ID 列表
        self.image_ids = image_ids
        # 图片的标签和基本信息
        self.data = data
        # 图像增强
        self.transforms = transforms
        # 测试集
        self.test = test

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]

        # 百分之 50 的概率会做 mix up
        if self.test or random.random() > 0.5:
            # 具体定义在后面
            image, boxes = self.load_image_and_boxes(index)
        else:
            # 具体定义在后面
            image, boxes = self.load_mixup_image_and_boxes(index)

        # 这里只有一类的目标定位问题，标签数量就是 bbox 的数量
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([index])

        # 多做几次图像增强，防止有图像增强失败，如果成功，则直接返回。
        if self.transforms:
            for i in range(10):
                sample = self.transforms(**{
                    'image': image,
                    'bboxes': target['boxes'],
                    'labels': labels
                })
                if len(sample['bboxes']) > 0:
                    image = sample['image']
                    target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
                    target['boxes'][:, [0, 1, 2, 3]] = target['boxes'][:, [1, 0, 3, 2]]  # yxyx: be warning
                    break

        return image, target, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]

    def load_image_and_boxes(self, index):
        # 加载 image_id 名字
        image_id = self.image_ids[index]
        # 加载图片
        image = cv2.imread(f'{self.image_root}/{image_id}.jpg', cv2.IMREAD_COLOR)
        # 转换图片通道 从 BGR 到 RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # 0,1 归一化
        image /= 255.0
        # 获取对应 image_id 的信息
        records = self.data[self.data['image_id'] == image_id]
        # 获取 bbox
        boxes = records[['x', 'y', 'w', 'h']].values
        # 转换成模型输入需要的格式
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        return image, boxes

    # 自定义 transform
    def load_mixup_image_and_boxes(self, index, imsize=1024):
        # 加载图片和 bbox
        image, boxes = self.load_image_and_boxes(index)
        # 随机加载另外一张图片和 bbox
        r_image, r_boxes = self.load_image_and_boxes(random.randint(0, self.image_ids.shape[0] - 1))
        # 进行 mixup 图片的融合，这里简单的利用 0.5 权重
        mixup_image = (image + r_image) / 2
        # 进行 mixup bbox的融合
        mixup_boxes = np.concatenate((boxes, r_boxes), 0)
        return mixup_image, mixup_boxes

# 取第 0 折为验证集，其余 4 折为训练集
fold_number = 0

train_dataset = WheatData(
    image_ids=df_folds[df_folds['fold'] == 2].index.values,
    data=data,
    transforms=get_train_transforms(),
    test=False,
)

validation_dataset = WheatData(
    image_ids=df_folds[df_folds['fold'] == fold_number].index.values,
    data=data,
    transforms=get_valid_transforms(),
    test=True,
)

# 验证一下生成器得到的训练数据是否正确
image, target, image_id = train_dataset[0]
boxes = target['boxes'].cpu().numpy().astype(np.int32)
numpy_image = image.permute(1, 2, 0).cpu().numpy()
fig, ax = plt.subplots(1, 1, figsize=(16, 8));
for box in boxes:
    cv2.rectangle(numpy_image, (box[1], box[0]), (box[3], box[2]), (0, 1, 0), 2);
ax.set_axis_off()
ax.imshow(numpy_image)
plt.show()