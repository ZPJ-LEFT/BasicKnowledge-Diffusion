# Scored-based Diffusion Model

如今提到扩散模型主要理解为以2020年DDPM（Denoising Diffusion Probabilistic Models）为开端及其后续改进的模型。但其实2019年宋飏率先提出了以得分为基础的生成扩散模型SMLD（Score Matching with Langevin Dynamics）。两者虽然在细节上有些许不同，不过都是基于噪声模型和逐步逼近真实数据分布的思想建立的，并且宋飏在2021年从一个更高的视角——随机微分方程（Stochastic Differential Equation, SDE）将DDPM和SMLD两者进行了统一（扩散模型框架——随机微分方程（SDE））。
