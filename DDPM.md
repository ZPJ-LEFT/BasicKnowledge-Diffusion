# [DDPM](https://zhuanlan.zhihu.com/p/666552214)

将图像定义为一个概率分布，用正态分布进行表征，生成图像的过程就是从概率分布中采样的过程。

扩散模型是一个隐变量模型，给定参数$\theta$，生成干净图像$x_{0}$的概率分布是$p_{\theta}=\int p_{\theta}(x_{0:T})dx_{1:T}$，其中$x1,...,xT$是隐变量

整个扩散过程被定义为一个马尔可夫链，即当前时刻的状态仅有上一时刻的状态决定

## 前向过程

从x0到xT，是不断加噪的过程，记为q

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

需要根据当前时刻的图片，预测前一时刻的图片：

$$
\mu = \frac{1}{ \sqrt{\alpha_t }}(x_{t} -  \frac{1-\alpha_t}{\sqrt{1-\overline{\alpha}_{t}}}\epsilon)
$$

$$
\sigma^{2} = \frac{\beta_{t}(1-\overline{\alpha}_{t-1})}{1-\overline{\alpha}_{t}}
$$
