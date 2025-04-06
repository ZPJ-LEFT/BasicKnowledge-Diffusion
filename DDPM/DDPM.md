# DDPM (Denoising Diffusion Probabilistic Models)

将图像定义为一个概率分布，生成图像的过程就是从概率分布中采样的过程。

扩散模型是一个隐变量模型，给定参数$\theta$，生成干净图像$x_{0}$的概率分布是：

$$
p_{\theta}(x_0)=\int p_{\theta}(x_{0:T})dx_{1:T}=\int p_{\theta}(x_{0:T})dx_1dx_2···dx_T
$$

其中$x_1,...,x_T$是隐变量，$x_0 \sim q(x_0)$是真实图像，$p_\theta(x_{0:T})$是$x_0,x_1,...,x_T$的联合概率密度，求解$x_0$的边缘概率分布就是对其他变量求积分，这是边缘概率密度的通用求法。

概率乘法公式：对于联合概率$P(A_1A_2···A_n)>0$，有:

$$
\begin{aligned}
P(A_1A_2···A_n) &= P(A_1A_2···A_{n-1})P(A_n|A_1A_2···A_{n-1})\\
&=P(A_1)P(A_2|A_1)P(A_3|A_1A_2)···P(A_n|A1···A_{n-1})
\end{aligned}
$$

整个扩散过程被定义为一个马尔可夫链，即当前时刻的状态仅由上一时刻的状态决定：

$$
\begin{aligned}
p_{\theta}(x_{0:T})&=p_{\theta}(x_0|x_1)p_{\theta}(x_1|x_2)···p_{\theta}(x_{T-1}|x_T)p(x_T) \\
&=p_\theta(x_n|x_{n-1})p_\theta(x_{n-1}|x_{n-2})···p_\theta(x_1|x_0)p(x_0)
\end{aligned}
$$

以上两个公式，分别代表前向加噪过程和后向去噪过程。

## 前向过程

从x0到xT，是不断加噪的过程，记为q：

$$
q(x_{1:T}|x_{0}) = \prod_{t=1}^{T}q(x_t|x_{t-1})
$$

$$
q(x_{t}|x_{t-1}) \sim N(x_{t};\sqrt{1-\beta_{t}}x_{t-1},\beta_{t}I)
$$

利用参数重整化（通过一个确定性函数和随机噪声，表达满足某种分布的随机变量），可以写出$x_{t}$如下：
$$
x_{t} = \sqrt{1-\beta_{t}} x_{t-1} + \sqrt{\beta_{t}}\epsilon_{t-1}, \, \, \, \epsilon_{t-1} \sim N(0,I)
$$

令$\alpha_{t}=1-\beta_{t}$, $\overline{\alpha}_{t}=\prod_{i=1}^{t}\alpha_{i}$，可以得出：

$$
x_{t} = \sqrt{\overline{\alpha}_{t}}x_{0}+\sqrt{1-\overline{\alpha}_{t}}\epsilon
$$

前向过程总结：

- 给定$x_0$和参数$\alpha$，可以直接生成$x_{T}$
- $\beta_{t}$随着t的增长逐渐增大
- 加噪过程将图像逐渐转变为均值为0，方差为I的纯噪声图像

## 反向过程

反向过程p表征为：

$$
p(x_{t-1}|x_{t}) \sim N(x_{t-1}; \mu_{\theta}(x_{t},t),\sigma^{2}_{\theta}(x_t,t))
$$

利用贝叶斯公式：

$$
p(x_{t-1}|x_{t}) = \frac{p(x_{t}|x_{t-1})p(x_{t-1})}{p(x_t)}
$$

额外加入条件$x_0$：

$$
\begin{aligned}
p(x_{t-1}|x_{t},x_0) &= \frac{p(x_{t}|x_{t-1},x_0)p(x_{t-1}|x_0)}{p(x_t|x_0)}\\
&=\frac{N(\sqrt{\alpha_{t}}x_{t-1}, (1-\alpha_{t})I) N(\sqrt{\overline{\alpha}_{t-1}}x_0,(1-\overline{\alpha}_{t-1})I)}{N(\sqrt{\overline{\alpha}_{t}}x_0,(1-\overline{\alpha}_{t})I)}\\
& \sim N(\mu, \sigma^2)
\end{aligned}
$$

展开后得到：

$$
\mu = \frac{\sqrt{\overline{\alpha}_{t-1}}(1-\alpha_t)}{1-\overline{\alpha}_t}x_0 + \frac{(1-\overline{\alpha}_{t-1})\sqrt{\overline{\alpha}_{t}}}{1-\overline{\alpha}_t}x_t
$$

$$
\sigma^2 = \frac{(1-\alpha_t)(1-\overline{\alpha}_t)}{1-\overline{\alpha}_{t}}
$$

将$x_0$替换为$x_{t}$可得：

$$
\mu = \frac{1}{ \sqrt{\alpha_t }}(x_{t} -  \frac{1-\alpha_t}{\sqrt{1-\overline{\alpha}_{t}}}\epsilon)
$$

$$
\sigma^{2} = \frac{\beta_{t}(1-\overline{\alpha}_{t-1})}{1-\overline{\alpha}_{t}}
$$

## 损失函数

基于变分推断，可以推导扩散模型的优化目标，将扩散模型视为隐变量模型，可以获得优化目标如下：

$$
\begin{aligned}
logp_\theta(x_0)&=log \sum p_\theta(x_{0:T})d_{x1:T}\\
&=log \sum \frac{p_\theta(x_{0:T})q(x_{1:T}|x_0)}{q(x_{1:T}|x_0)}d_{x1:T}\\
&\geq E_{q(x_{1:T|x_0})}log\frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)} \\
\end{aligned}
$$

优化目标为上式取负值：

$$
\begin{aligned}
L &= E_{q(x_{1:T|x_0})}log\frac{q(x_{1:T}|x_0)}{p_\theta(x_{0:T})} \\
&= \,\,···\\
&= D_{KL}(q(x_T|x_0)||p_\theta(x_T)) + \sum_{t=2}^T E_{q(x_t|x_0)}D_{KL}[q(x_{t-1}|x_t,x_0)||p_\theta(x_{t-1}|x_t)]-logp_\theta(x_0|x_1)
\end{aligned}
$$%

上式中的三项，第一项可以省略，因为$p(x_T)$和$q(x_T|x_0)$都被定义为高斯分布，第三项可以构造一个高斯重建误差，但通常可以省略。

因此损失函数的重点是第二项：

$$
D_{KL}[q(x_{t-1}|x_t,x_0)||p_\theta(x_{t-1}|x_t)]
$$

假定$q(x_{t-1}|x_t,x_0)$也是高斯分布，并且方差为固定项。

$$
p(x_{t-1}|x_{t}) \sim N(x_{t-1}; \mu_{\theta}(x_{t},t),\sigma^{2}_t)
$$

此时优化目标为:

$$
L = E_{q(x_t|x_0)}[\frac{1}{2\sigma^{2}_t}||\mu_t(x_t,x_0)-\mu_\theta(x_t,t)||^2]
$$

但在实践中，通常不选择预测均值，将均值公式进一步展开可得：

$$
L = E_{x_0,\epsilon \sim N(0,I)}[\frac{\beta_t^2}{2\sigma_{t}^2 \alpha_t (1-\overline{\alpha}_t)}||\epsilon-\epsilon_\theta(x_t,t)||^2]
$$\

去掉权重系数，并且展开$x_t$后:

$$
L^{simple} = E_{x_0,\epsilon \sim N(0,I)}[||\epsilon-\epsilon_\theta(\sqrt{\overline{\alpha}_{t}}x_{0}+\sqrt{1-\overline{\alpha}_{t}}\epsilon,t)||^2]
$$

从DDPM的对比实验结果来看，预测噪音比预测均值效果要好，采用简化版本的优化目标比VLB目标效果要好。

## 模型训练与推理

虽然扩散模型背后的推导比较复杂，但是我们最终得到的优化目标非常简单，就是让网络预测的噪音和真实的噪音一致。

### 模型训练

1. 读取一个原始图像作为训练数据：$x_0 \sim q(x_0)$
2. 随机选择一个噪声步： $t \sim Uniform(\{1,...,T\})$
3. 生成一个高斯噪声：$\epsilon \sim N(0,I)$
4. 梯度下降：$\triangledown_\theta||\epsilon-\epsilon_\theta(\sqrt{\overline{\alpha}_t}x_0+\sqrt{1-\overline{\alpha}_t}\epsilon,t)||^2$

### 模型推理

1. 初始化纯噪声图像：$x_T \sim N(0,I)$
2. for t = T,...,1 do
    - 生成一个高斯噪声：$z \sim N(0,I)$
    - $x_{t-1} = \frac{1}{ \sqrt{\alpha_t }}(x_{t} -  \frac{1-\alpha_t}{\sqrt{1-\overline{\alpha}_{t}}}\epsilon_\theta(x_t,t))+\sigma_t z$
3. end for

## 网络设计

目前主流设计思路是：U-Net + 时间编码 + 自注意力 + 残差连接。

- 时间编码（Time Embedding）
- Unet架构
- 残差块
- 注意力机制
- GroupNorm+Swish

### Pytorch代码实现

略

# 参考文献

- [扩散模型(Diffusion Model)奠基之作：DDPM 论文解读](https://zhuanlan.zhihu.com/p/682840224)
- [一文带你看懂DDPM和DDIM（含原理简易推导，pytorch代码）](https://zhuanlan.zhihu.com/p/666552214)
- [扩散模型之DDPM](https://zhuanlan.zhihu.com/p/563661713)