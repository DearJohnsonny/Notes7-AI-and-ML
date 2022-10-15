<a href="https://dearjohnsonny.github.io/Notes1-Biotech/">Notes1-Biotech</a>

<a href="https://dearjohnsonny.github.io/Notes2-Biotech/">Notes2-Biotech</a>

<a href="https://dearjohnsonny.github.io/Notes3-Statistics/">Notes3-Statistics</a>

<a href="https://dearjohnsonny.github.io/Notes4-Linear-Algebra/">Notes4-Linear-Algebra</a>

<div align=center>
<img src="https://user-images.githubusercontent.com/111955215/195480902-bbd1fc80-9665-437f-9f43-340a2189e35a.png" width="900">
</div>

# 基础知识

## 基本的术语
**标签**（Lable）是指我们要预测的内容，即简单线性回归中的 y 变量。

**特征**（Features）是输入变量，即简单线性回归中的 x 变量。

**样本**（Examples）是指数据的特定实例：x。（我们将 x 显示为粗体，表示它是一个矢量。）我们将示例分为两类：
* 有标签样本同时包含特征和标签。具体来说：labeled examples: {features, label}: (x, y)
* 无标签样本包含特征，但不包含标签。具体来说：  unlabeled examples: {features, ?}: (x, ?)

**模型**（Models）定义了特征和标签之间的关系。例如，垃圾内容检测模型可能会将某些功能与“垃圾内容”紧密关联。我们重点介绍模型生命周期的两个阶段：
* 训练是指创建或学习模型。也就是说，您向模型展示有标签样本，让模型逐渐学习特征与标签之间的关系。
* 推断表示将经过训练的模型应用于无标签样本。也就是说，使用经过训练的模型做出有用的预测 (y')。例如，在推理期间，您可以针对新的无标签样本预测 medianHouseValue。

**回归与分类（这俩都是监督学习）**

**回归模型**可预测连续值。例如，回归模型做出的预测可回答如下问题：
* 加利福尼亚州一栋房子的价值是多少？
* 用户点击此广告的可能性有多大？

**分类模型**可预测离散值。例如，分类模型做出的预测可回答如下问题：
* 指定的电子邮件是垃圾邮件还是非垃圾邮件？
* 这是狗、猫还是仓鼠的图片？


## 机器学习算法
Supervised Learning 监督学习：学习数据带有标签

Unsupervised Learning 无监督学习：没有任何的标签，或者有相同的标签。已知数据集，不知如何处理，也未告知每个数据点是什么。

<div align=center>
<img src="ttps://user-images.githubusercontent.com/111955215/195987501-e98862ee-d5bc-48ef-8cb3-703c837a94dd.png" width="900">
</div>

右侧的例子，无监督学习将数据划分为两个集合，也就是聚类clustering algorithm
