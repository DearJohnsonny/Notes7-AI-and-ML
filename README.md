<a href="https://dearjohnsonny.github.io/Notes1-Biotech/">Notes1-Biotech</a>

<a href="https://dearjohnsonny.github.io/Notes2-Biotech/">Notes2-Biotech</a>

<a href="https://dearjohnsonny.github.io/Notes3-Statistics/">Notes3-Statistics</a>

<a href="https://dearjohnsonny.github.io/Notes4-Linear-Algebra/">Notes4-Linear-Algebra</a>

<div align=center>
<img src="https://user-images.githubusercontent.com/111955215/195480902-bbd1fc80-9665-437f-9f43-340a2189e35a.png" width="900">
</div>

<div align=center>
<img src="https://user-images.githubusercontent.com/111955215/199656486-321639ab-85f0-4d98-ba62-00c111f04142.png" width="900">
</div>

# 基础知识
## 关于模型
计算学习模型 computational learning theory
PAC 概率近似正确： $P(|f(\boldsymbol{x})-y| \leq \epsilon) \geq 1-\delta$

### 评估方法

#### Holdout方法验证
holdout 方法将初始数据集分成单独的训练集（Training Set）和测试集（Test Set），前者用于模型训练，后者用于评估模型的泛化性能。在典型的机器学习程序中，人们会对超参数进行不断调整和比较，以进一步提高对不可见数据进行预测的性能。这个过程被称为模型选择（Model Selection），尝试将超参数调整至最优。

但是，如果在模型选择过程中重复使用相同的测试集，那么这个测试集就不是真正意义上的测试集。真正的测试集应该是不可见的。如果一直尝试对测试集拟合，这个测试集实际上就是训练集的一部分。

holdout 方法的缺点是算法评估对数据划分非常敏感，对于不同的数据划分比例和不同的分布，评估结果可能会有较大差异

#### 保持交叉验证
使用训练集和验证集进行反复训练并将超参数调整至较优水平，再使用测试集来评估模型的泛化性能：

<div align=center>
<img src="https://user-images.githubusercontent.com/111955215/199655382-ac537220-41ba-4ccb-b098-b836e48675af.png" width="900">
</div>

#### K-折交叉验证 K-Fold Cross Validation
K 折交叉验证的步骤是，随机地将训练数据集分成 K 次折叠。其中 K-1 次折叠用于模型训练、另外一个折叠用于性能评估，上述步骤重复 K 次（每次抽取不同的折叠），获得 K 个模型的性能评估结果。

K 折交叉验证可以得到令人满意的泛化性能的最优超参数值，具有更高的准确率和鲁棒性。K 折交叉验证表现好的原因在于其拥有更多的训练样本，且每个训练样本都恰好验证一次，这样可以产生较低的方差。

<div align=center>
<img src="https://user-images.githubusercontent.com/111955215/199655395-d257b4b8-de44-410a-96d8-3b9f468a8454.png" width="900">
</div>

## 一些思考
### 三个关键问题
如何获得测试结果? —— 评估方法

如何评估性能优劣? —— 性能度量

如何判断实质差别? —— 比较检验


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

**泛化误差**：在 “末来”样本上的误差

**经檢误差**: 在训练集上的误差, 亦称 “训练误差”

过拟合 (Overfitting) ：模型学习能力过墙以至于将实验者不希望的特征也学习了，学到了不该学的东西。U型曲线的右侧

欠拟合 (Underfitting)。U型曲线的左侧

<div align=center>
<img src="https://user-images.githubusercontent.com/111955215/199653715-0a6f1f80-6bed-4741-a67a-c0cd4f793cec.png" width="900">
</div>

- 数据集; 训练, 测试
- 示例(instance), 样例(example)
- 样本(sample)
- 属性(attribute) = 特征(feature)。其实就是feature中的一个维度，其值为属性值。
- 属性空间 = 样本空间 = 输入空间。**可以理解为定义域**
- 特征向量(feature vector)，也就是因变量的各个维度
- 标记空间,输出空间。**可以理解为值域**

机器学习的前提：
- 未见样本(unseen instance)
- 未知 "分布"
- 独立同分布(i.i.d.)：否则无法利用概率统计的工具
- 泛化(generalization)：训练出来的模型处理新数据的能力，可以理解为推广。对新数据的处理能力越强，则泛化性越好，也就是 $P(|f(\boldsymbol{x})-y| \leq \epsilon) \geq 1-\delta$ 中的   $\epsilon$ 比较小
## 机器学习算法
### Supervised Learning 监督学习
监督学习：学习数据带有标签

在监督式学习中，机器学习算法通过检查许多示例并尝试找到将损失降至最低的模型来构建模型；此过程称为经验风险最小化。损失是错误预测的惩罚。也就是说，损失是一个表示模型在单个样本上的预测质量的数字。如果模型的预测完全准确，则损失为零，否则损失会更大。训练模型的目的是从所有样本中找到一组平均损失“较小”的权重和偏差。

用**损失函数**（**Loss function**）来对模型的结果进行判断

线性回归模型使用一种称为**平方损失函数**（也称为 L2 损失）的损失函数

<div align=center>
<img src="https://user-images.githubusercontent.com/111955215/199656174-c3b5fcb4-534c-4679-97b3-2803f9a147d7.png" width="700">
</div>

#### 分类(Classification)
在分类中，函数的输出是一个分类标签。常见的例子有垃圾邮件识别和手写数字识别。

<div align=center>
<img src="https://user-images.githubusercontent.com/111955215/199656055-a970304b-42e8-4652-99b7-f858b0595387.png" width="400">
</div>

#### 回归(Regression)
在回归中，函数的输出可以是一个连续的值，通常是一个数学函数。例如，对于某一职位，员工薪水和工龄之间的关系。

<div align=center>
<img src="https://user-images.githubusercontent.com/111955215/199656104-0e42b15b-908f-4dba-a0ca-36da46899828.png" width="400">
</div>

### Unsupervised Learning 无监督学习
无监督学习：没有任何的标签，或者有相同的标签。已知数据集，不知如何处理，也未告知每个数据点是什么。

#### 聚类 (Clustering)
聚类是一种探索性数据分析技术，能够在不知道输入对象的任何先验知识的情况下，将其分成不同的集群。每个集群包含一定程度相似的对象，但是与其他集群中的对象不相似。例如营销人员将客户分类成不同的客户群。

<div align=center>
<img src="https://user-images.githubusercontent.com/111955215/199656342-319e09b7-06bb-4622-bf2b-e19658599e18.png" width="400">
</div>

#### 降维 (Dimensionality Reduction)
通常，我们正在处理高维度的数据，这对于有限的存储空间和机器学习算法的计算性能是很大的挑战。降维是指在某些限定条件下，降低随机变量的维度，得到一组“不相关”的主变量，同时保留大部分有效信息。

<div align=center>
<img src="https://user-images.githubusercontent.com/111955215/199656324-f19cf2a8-84e7-4694-99f5-05fb258442e8.png" width="400">
</div>

### MCP神经元和Rosenblatt 阈值感知器
MCP 神经元为具有二进制输出的简单逻辑门。多个信号从树突传入细胞体中，如果累积输入信号超过一定阈值，则从轴突传出输出信号。

感知器采用输入向量 $x$ 和权重向量 $w$ ，净输入 $z$ 为这二者的线性组合 $\left(z=w_1 x_1+\ldots+w_m x_m\right)$ :

$$
w=\left[\begin{array}{c}
w_1 \\
w_2 \\
\vdots \\
w_m
\end{array}\right], x=\left[\begin{array}{c}
x_1 \\
x_2 \\
\vdots \\
x_m
\end{array}\right]
$$
激活函数采用阶跃函数 $\phi(z), \theta$ 为神经元的触发阈值:
$$
\phi(z)= \begin{cases}1, & \text { if } z \geq \theta \\ -1, & \text { if } z<\theta\end{cases}
$$
通过设定 $x_0=1$ 和 $w_0=-\theta$ ，将 $\theta$ 集成到 $z$ 中，可以让式更整齐:
$$
\begin{gathered}
z=w_0 x_0+w_1 x_1+w_2 x_2+\ldots+w_m x_m=w^T x \\
\phi(z)= \begin{cases}1, & \text { if } z \geq 0 \\
-1, & \text { if } z<0\end{cases}
\end{gathered}
$$

$$
\text { 下图为通过 } z=w^T x \text { 得到 } 1 \text { 或 - } 1 \text { 的激活函数 (左子图)，以及其用于区分两个线性可分的类（右子图）: }
$$


<div align=center>
<img src="https://user-images.githubusercontent.com/111955215/199718349-f73e0ba5-49c6-4ab7-94a7-f2bb1774b607.png" width="400">
</div>

Rosenblatt 阈值感知器就看下面这个<a href="https://www.szdev.com/blog/AI/logistic-regression-mathmatics-and-vectorization/">原始帖子</a>就好

### 对率回归 Logistic regression
Adaline 和 逻辑回归的差异，如下所示：

Adaline:
* 激活函数：线性函数 $\phi(z)=z$
* 阈值函数: 阶跃函数 $\hat{y}= \begin{cases}1 & \text { if } \phi(z) \geq 0 \\ 0 & \text { otherwise }\end{cases}$
逻辑回归:
* 激活函数: sigmoid 函数 $\phi(z)=\frac{1}{1+e^{-z}}$
* 阈值函数：阶跃函数 $\hat{y}= \begin{cases}1 & \text { if } \phi(z) \geq 0.5 \\ 0 & \text { otherwise }\end{cases}$
需要注意的是，逻辑回归中的条件 $\phi(z) \geq 0.5$ ，等价于 $z \geq 0$ 。

<div align=center>
<img src="https://user-images.githubusercontent.com/111955215/199717381-0245b62c-1d5b-484f-9900-3bd7a9806833.png" width="400">
</div>

## 降低损失
“模型”接受一个或多个特征作为输入，并返回一个预测结果 $\left(y^{\prime}\right)$ 作为输出。用损失函数（如平方损失函数）测试结果，并生成新的模型参数。学习过程会持续迭代，直到算法发现损失可能最低的模型参数。通常，系统会不断迭代，直到整体损失停止变化或至少变化非常缓慢。如果发生这种情况，我们会说模型已收敛。

<div align=center>
<img src="https://user-images.githubusercontent.com/111955215/195988287-6fdb10c4-5785-4fa5-8f24-484563cded14.png" width="900">
</div>
