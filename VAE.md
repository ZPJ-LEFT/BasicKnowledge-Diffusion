[参考文章](https://link.zhihu.com/?target=https%3A//www.cnblogs.com/amazingter/p/14686450)

## AE(AutoEncoder)
自编码器，用于学习无标签数据的有效编码，属于无监督学习。

AutoEncoder的目的：学习高维数据的低维表示，常用于降维、压缩。

训练该模型的目标是让输入和输出的误差尽量小，损失函数为:

$$
L = x - d(e(x))
$$

AE中的隐变量不具备语义特征，也不具有随机性。

缺点：只能用于编码解码，无法用于生成。

## VAE(Variational AutoEncoder)
变分自编码器=AE+概率生成模型，在隐空间内引入概率分布，使模型能生成具有多样性的样本，并且在学习过程中能更好地理解数据分布。

### 代码

```
class VAE(nn.Module):
    def __init__(self, input_dim=256, hiddent_dim=64, latent_num=2, latent_dim=32, device='cpu'):
        self._device = device

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(0.2)
        )

        self.mean_layer = nn.Linear(latent_dim, latent_num)
        self.var_layer = nn.Linear(latent_dim, latent_num)

        self.generation_layer = nn.Sequential(
            nn.Linear(latent_num, latent_dim),
            nn.LeakyReLU(0.2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_num, hiddent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def enocde(self,x):
        x = self.encoder(x)
        mean = self.mean_layer(x)
        var = self.var_layer(x)
    
    def sampling(self, mean, var):
        epsilon = torch.rand_like(var).to(self._device)
        z = mean + var * epsilon
    
    def decode(self, x):
        x = self.generation_layer(x)
        return self.decoder(x)

    def forward(self, x):
        mean, var = self.encode(x)
        z = self.sampling(mean, var)
        x_hat = self.decoder(z)
        return x_hat, mean, var
```

损失函数：


# 参考文章

[文生图模型演进：AE、VAE、VQ-VAE、VQ-GAN、DALL-E 等 8 模型](https://mp.weixin.qq.com/s?__biz=Mzk0ODU3MjcxNA==&mid=2247485323&idx=1&sn=4408ac639f54f87c62cb64503cc2e9d9&chksm=c364c0cef41349d8f7a0c2d388b3de7bdfef049c8024b09e382e20a8e337e7c7acbca7b0a8e7&scene=178&cur_album_id=3154744884457668609#rd)

[【多模态】AE、VAE、VQVAE、VQGAN原理解读](https://zhuanlan.zhihu.com/p/657857297)