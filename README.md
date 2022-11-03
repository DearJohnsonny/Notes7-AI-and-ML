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

泛化误差：在 “末来”样本上的误差
经檢误差: 在训练集上的误差, 亦称 “训练误差”

## 机器学习算法
### Supervised Learning 监督学习
监督学习：学习数据带有标签

在监督式学习中，机器学习算法通过检查许多示例并尝试找到将损失降至最低的模型来构建模型；此过程称为经验风险最小化。损失是错误预测的惩罚。也就是说，损失是一个表示模型在单个样本上的预测质量的数字。如果模型的预测完全准确，则损失为零，否则损失会更大。训练模型的目的是从所有样本中找到一组平均损失“较小”的权重和偏差。

用**损失函数**（**Loss function**）来对模型的结果进行判断

线性回归模型使用一种称为**平方损失函数**（也称为 L2 损失）的损失函数

### Unsupervised Learning 无监督学习
无监督学习：没有任何的标签，或者有相同的标签。已知数据集，不知如何处理，也未告知每个数据点是什么。

<div align=center>
<img src="https://user-images.githubusercontent.com/111955215/195987501-e98862ee-d5bc-48ef-8cb3-703c837a94dd.png" width="900">
</div>

右侧的例子，无监督学习将数据划分为两个集合，也就是聚类clustering algorithm

## 降低损失
“模型”接受一个或多个特征作为输入，并返回一个预测结果 $\left(y^{\prime}\right)$ 作为输出。用损失函数（如平方损失函数）测试结果，并生成新的模型参数。学习过程会持续迭代，直到算法发现损失可能最低的模型参数。通常，系统会不断迭代，直到整体损失停止变化或至少变化非常缓慢。如果发生这种情况，我们会说模型已收敛。

<div align=center>
<img src="https://user-images.githubusercontent.com/111955215/195988287-6fdb10c4-5785-4fa5-8f24-484563cded14.png" width="900">
</div>
