# SDE扩散生成模型

核心论文：Score-Based Generative Modeling through Stochastic Differential Equations

相比于以往生成模型的做法，这篇 paper 的方法比较“数学&物理 style”—— 使用 SDE(随机微分方程) 来对图像生成任务进行建模。概括起来，这种玩法就是利用 SDE 将复杂的图像分布“平滑地过渡”到简单的先验分布(比如将一张图映射为标准高斯噪声)；同时利用 SDE 的逆向形式("reverse-time SDE")将先验分布映射回原来的图像分布。

这两个互逆的 SDE 过程实际在做的事情就是加噪与去噪，这点与 SMLD(Score Matching with Langevin Dynamics, 其实这里指的是 NCSN) 和 DDPM 的套路一样，但通过 SDE 这种视角来建模，在凸显高级感(笑)的同时，还能顺带将 score-based models 和 diffusion models 这两种生成模型的建模方式纳入到 SDE 这个统一的框架下。

特别地，作者还分别为 SMLD 和 DDPM 所对应的 SDE 形式起了名字，前者名曰 "Variance Exploding(VE) SDE"；后者名为 "Variance Preserving(VP) SDE"。另外，在 VP SDE 的基础上，作者还额外提出了一种叫 "sub-VP SDE" 的形式，它通常能够取得更好的采样质量和更高的似然(likelihood)，可以认为其是 VP SDE 的改进版。


## 什么是SDE？为什么要用SDE？

**SDE**是描述随机过程的工具之一。

以往像 SMLD 和 DDPM 这些老流派在训练时加噪的方式是在每个时间步对图像施加高斯噪声：

$$
\begin{cases}
DDPM\: x_i = \sqrt{1-\beta_i}x_{i-1} + \sqrt{\beta_i}\epsilon, \,\, \epsilon \in N(0,1)\\
NCSN: x_i = x_{i-1} + \sqrt{\sigma_i^2-\sigma_{i-1}^2}\epsilon, \,\, \epsilon \in N(0,1)\\
\end{cases}
$$

从物理学的角度看，以上过程本质上是离散时间$t \in [0:T]$内的扩散过程。

在扩散过程中，对于t时间点的图像$x_t$，考虑两种情况：
1. 当t固定时，$x_t$是随机变量，即$x_t \sim N(\sqrt{\overline{\alpha}_t}x_0, (1-\overline{\alpha}_t)I)$
2. 当x固定时，$x_t$是一个采样轨迹：$x_T,x_{T-1},...,x_0$。因此$x_t$实际上是一个随机过程。

考虑diffusion的离散采样过程，可以表示为：

$$
x_0\rightarrow x_1\rightarrow ··· x_t\rightarrow x_{t+1}\rightarrow ···\rightarrow x_T
$$

将该离散过程连续化，我们设$t\in[0,T] $，上述离散的前向加噪过程连续化，可以表示为：

$$
x_t\rightarrow x_{t+\Delta t}, \,\,\,\Delta t\rightarrow 0
$$

反向去噪过程表示为：
$$
x_{t+\Delta t}\rightarrow x_t, \,\,\,\Delta t\rightarrow 0
$$

## SDE-based Forward Process
对于图像x，用伊藤过程(Itô SDE)来建模连续的微分过程（扩散过程）为：

$$
dx = f(x,t)dt + g(t)dw
$$

其中：
- f(x,t): dirft coefficience （漂移系数）
- g(t): diffusion coefficient （Diffusion系数）
- w：brown motion （布朗运动）

也就是说，一次扩散dx是由确定性和不确定性两个过程组合而成。

布朗运动具有增量独立性、增量服从高斯分布且轨迹连续。我们定义f(x,t)dt为确定性变化过程，g(t)dw为不确定性变化过程(dw-布朗运动的微分就相当于随机采样)。

PS：上式是伊藤 SDE 的通用表达形式，在实际建模过程中，可以设置不同的漂移系数和扩散系数，SMLD 和 DDPM 就恰好是两种不同 SDE 的数值离散形式。

## SDE-based Reverse Process

在扩散过程中时间变量是$0\rightarrow T$正向递增的，对应建模的是网络的训练过程，这个过程将图像变成纯噪声。那么，推理(采样)过程就应该是它的逆过程，从而将纯噪声“变回”像原图那样有意义的图像。

基于[已有的研究](https://www.sciencedirect.com/science/article/pii/0304414982900515)，伊藤过程的逆过程也是一个伊藤过程，其SDE表达式为：

$$
dx = [f(x,t)-g(t)^2\triangledown_x log p_t(x)]dt + g(t)d\overline{w}
$$

其中$\overline{w}$表示在反向过程中的布朗运动方向，dt表示微小的负时间步长(请注意，这个逆向 SDE 的时间“流动方向”是 
$T\rightarrow 0$，因此这里 $dt$
 是绝对值无穷小的负值)。

在我们已知前向过程的漂移、扩散系数以及
score function时，可以通过数值求解反向SDE，从而求解$x_0$。

## 学习目标



# 参考文献
- [扩散模型 | 2.SDE精讲](https://zhuanlan.zhihu.com/p/677154173)
- [Score-based SDE 扩散生成模型从入门到出师系列(一)：用随机微分方程建模图像生成任务并统一分数和扩散模型](https://zhuanlan.zhihu.com/p/689276382)