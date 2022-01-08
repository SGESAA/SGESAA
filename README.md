# SGESAA
基于自引导进化策略的高效数据增强算法
## 算法介绍
将自动化数据增强问题转换为连续化策略向量的搜索问题，并且使用自引导进化策略的策略向量搜索方法，通过引入历史估计梯度信息指导探索点的采样与更新，在能够有效避免陷入局部最优解的同时，提升搜索过程的收敛速度。
## 基本原理
![原理图](/images/SGESAA算法流程.png)
## 使用方法
### 环境配置
```python
pip install -r requirements.txt
```
### 运行 SGES AA
```python
python autoaugment/tasks/images_classification/ars_search.py -c autoaugment/tasks/images_classification/confs/sges.yaml autoaugment/domain/vision/classification/confs/search_resnet18_cifar10.yaml
```
在数据增强任务结束之后，需要将输出的数据增强策略复制粘贴到autoaugment/domain/vision/classification/archive.py的sges_policy函数中
### 评估数据增强策略
比如进行```cifar10```数据集上```WRN40x2```架构的实验
``` python
CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore -u -m torch.distributed.launch --nproc_per_node=4 autoaugment/domain/vision/classification/train.py -c autoaugment/domain/vision/classification/confs/wrn_40x2_cifar10.yaml --verbose --aug=sges
```
