# SMLD (Score Matching Langevin Dynamics)

核心论文：Generative Modeling by Estimating Gradients of the
 Data Distribution

它估计 **数据分布相关的梯度** 并基于 **朗之万动力学(Langevin Dynamics)** 的思想来生成样本。

## 关于Score

### 什么是Score？

生成模型的一般定义是：给定从真实分布$p(x)$采样的观测数据$x \sim p(x)$，训练得到一个由$\theta$控制逼近真实分布的$p_\theta(x)$，称$p_\theta(x)$为生成模型。

极大似然估计的做法是极大化对数似然$logp_\theta(x)$，但在scored-based模型中没有采用该方法，而是定义得分函数(score function)：

$$
s_\theta(x) = \triangledown_x logp_\theta(x)
$$

score function是一个“矢(向)量场”(vector field)，它的方向表示：对于输入数据(样本)来说，其对数概率密度增长最快的方向。

极大似然估计目标是学习数据的**真实分布**，而Scored-based方法目标是学习数据的**形状和走势**。

### 为什么用Score？

首先要想明白最终生成的样本应该达到怎样的效果：在保证“创新性”的同时也要符合(靠近)原数据分布。
- 不能和原来的数据一模一样
- 也不能随意生成

结合上一节的解析，我们可以启发式地想到：

如果在采样过程中沿着分数的方向走，就能够走到数据分布的高概率密度区域，最终生成的样本就会符合原数据分布。

## 利用Score Matching求解Score Funcion

我们希望训练一个神经网络$\theta$使得分函数$s_\theta (x)$与真实梯度$\triangledown_x logp(x)$尽可能接近，即最小化模型分布和真实分布的Fisher散度：

$$
L=\frac{1}{2}E_{p(x)}[||\triangledown_x logp(x)-s_\theta (x)||^2]
$$

但实际上$p(x)$是未知的，这里提供一种无需知道$p(x)$也能计算L的方法，将上式进行展开二次项：

$$
L=\frac{1}{2} \int p(x) [||\triangledown_x logp(x)||^2 + ||s_\theta (x)||^2-2(\triangledown_x logp(x))^Ts_\theta (x)] dx\
$$

其中：
1. $||\triangledown_x logp(x)||^2$是常数项，**可忽略**
2. $||s_\theta (x)||^2$是模型的输出结果
3. $-2(\triangledown_x logp(x))^Ts_\theta (x)$需要继续推导，最终表示为：$\int p(x) [-2(\triangledown_x logp(x))^Ts_\theta (x)] dx
= 2\int p(x)[tr(\triangledown_x s_\theta(x))]dx$

基于以上推导，损失函数可以写为：

$$
L=\int p(x)[\frac{1}{2}||s_\theta(x)||^2+tr(\triangledown_x s_\theta(x))]dx
$$

其中$||s_\theta(x)||^2$是模型近似的结果，$\triangledown_x s_\theta(x)$是关于x的偏导，$tr(·)$表示矩阵的迹。

但是，对于高维数据而言，$tr(\frac{d(s_\theta(x))}{dx})$需要经历多次反向传播求偏微分，为解决该问题，有两种解决方法：sliced score matching & **denoising score matching**。

## 郎之万动力学采样(Sampling with Langevin Dynamics)
我们已经根据Score Matching的$argmin_\theta L$，得到score function $s_\theta(x)=\triangledown_x logp(x)$，接下来讨论如何基于得分函数进行采样生成。

公式表示如下：
$$
x_t = x_{t-1} + \triangledown_{x_{t-1}}logp(x+{t-1})
$$

但是该式与梯度下降法完全一致，但是我们不希望它最终收敛到某几个固定位置，而是希望它的结果具有多样性，因此利用郎之万动力学引入随机性：

$$
x_t = x_{t-1} + \frac{\epsilon}{2}\triangledown_{x_{t-1}}logp(x+{t-1}) + \sqrt{\epsilon}z_t
$$

其中$z_t \in N(0,1)$，$\epsilon$是预先定义的步长，从理论上看，假设T是最终时刻，当$T \rightarrow \infty$并且$\epsilon \rightarrow 0$时，最终生成的样本$x_T \sim p(x)$

## 分数模型的“困局”

分数模型在实际应用时会面临一些困难。概括起来，主要是以下三点：

- loss 不收敛
- 模型估计的分数不准
- 生成结果与原数据分布偏差大

这些问题的根源主要在于“流形假设”（manifold hypothesis）：流形假设指出，现实世界的数据分布倾向于聚集在内嵌于一个高维空间（也叫 ambinet space）的低维流形上。这个假设对于大部分数据集（自然也包括这里用来训练 score-based model 的数据集）都是成立的，它也是流形学习的理论基础。

但是score function是定义在整个高维空间上的，当x被限制在一个低维流形时，score是undefined，所以在低概率密度区域，score的预测是不准确的。

## NCSN (Noise Conditional Score Network)

为了解决分数模型的问题，作者提出NCSN，用高斯噪声去扰动数据，估计不同噪声扰动下数据分布的score。

由此，在高噪声扰动下，数据分布可以覆盖整个概率空间，弥补原始数据分布范围有限的问题；同时随着采样次数的增加，噪声扰动逐渐减小，score接近真实数据分布，确保最终采样结果与真实分布相似。

### 噪声设计原则

高噪声扰动下，能有效“填充”概率密度低的区域，缓解score估计不准确的问题，但同时会导致加噪后的数据分布偏离原始数据分布；低噪声扰动下，虽然能保证加噪后数据分布的一致性，但是不能很好地“填充”低密度区域。

为了权衡二者，可以利用多尺度噪声，设置L个标准差递增的高斯噪声，$\sigma_1 < \sigma_2<···<\sigma_L$，在训练过程中，先用$N(0,\sigma_i^2I)$对数据分布$p_{data}(x)$进行扰动，得到新的数据分布：

$$
p_{\sigma_i}(x) = \int p_{data}(y)N(x
|y,\sigma_i^2I)dy
$$

我们很容易从这个分布中采样出新的训练样本——只需要先从原数据分布中采样一个样本$x\sim p_{data}(x)$，然后加上相应高斯噪声，即可得到新的样本：

$$
x+\sigma_i z, \, z\sim N(0,I)
$$

### 训练方法

目标函数为:

$$
\sum_{i=1}^L \lambda(i) E_{p_{\sigma_i}(x)}[||\triangledown_x logp_{\sigma_i}(x) - s_{\theta}(x, \sigma_i) ||^2]
$$

其中$\lambda(i)$用于平衡不同噪声分布下的损失共享，一般取$\lambda(i) = \sigma_i^2$。

### Denoising Score Matching

回归到原始的损失，我们的未知项仅为$p_{data}(x)$，如果能明确知晓该分布的概率密度函数，即可求解损失。

对原始图像加噪后，图像分布满足高斯分布，$q_\sigma(\tilde{x}|x) \sim N(\tilde{x}|x,\sigma^2I)$，有论文证明:

$$
E_{q_\sigma(\tilde{x})}[||s_\theta(\tilde{x},\sigma)-\triangledown_xlogp_\sigma(\tilde{x})||^2] = E_{q_\sigma(\tilde{x}|x)}[||s_\theta(\tilde{x},\sigma)-\triangledown_xlogp_\sigma(\tilde{x}|x)||^2]
$$

等号左边是加噪后最直接的分数匹配形式，不妨称其为 显式的分数匹配(Explicit Score Matching)；而等号右边则是实际使用的去噪分数匹配形式(DSM)。

进一步计算得到:

$$
\triangledown_x log(q_\sigma(\tilde{x}|x)) = -\frac{\tilde{x}-x}{\sigma^2}
$$

最终得到：

$$
L(\theta;\sigma)=\frac{1}{2}E_{p_{data}(x)}E_{\tilde{x} \sim N(x,\sigma^2I)}[||s_\theta(\tilde{x}, \sigma)+\frac{\tilde{x}-x}{\sigma^2}||^2]
$$

$$
L(\theta;{\sigma_i}_{i=1}^L)=\frac{1}{L} \sum_{i=1}^L \lambda(i) L(\theta;\sigma_i)
$$

### 郎之万采样步长

在噪声强度$\sigma_i$下的采样步长设置为$\alpha_i \propto \sigma_i^2$，这是为了固定信噪比的量级，也就是说噪声越大，步长越大，作者设置是：

$$
\alpha_i = \epsilon \cdot \frac{\sigma_i^2}{\sigma_L^2}
$$

# 参考文献
- [Score-based生成扩散模型（一）——SMLD](https://zhuanlan.zhihu.com/p/718817935)
- [图像生成别只知道扩散模型(Diffusion Models)，还有基于梯度去噪的分数模型](https://zhuanlan.zhihu.com/p/597490389)
- [DiffusionModel-NCSN原理与推导](https://zhuanlan.zhihu.com/p/670052757)