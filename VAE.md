# VAE系列

- [VAE系列](#vae系列)
  - [AE(AutoEncoder)](#aeautoencoder)
  - [VAE(Variational AutoEncoder)](#vaevariational-autoencoder)
    - [代码](#代码)
  - [VQ-VAE(Vector Quantised VAE)](#vq-vaevector-quantised-vae)
    - [Vector Quantization技术](#vector-quantization技术)
    - [模型结构](#模型结构)
    - [模型训练](#模型训练)
    - [代码实现](#代码实现)
    - [Loss函数](#loss函数)
    - [图像生成](#图像生成)
  - [VQ-GAN](#vq-gan)
    - [重建模型](#重建模型)
    - [生成模型](#生成模型)
- [参考文章](#参考文章)


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
        self.std_layer = nn.Linear(latent_dim, latent_num)

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
        std = self.std_layer(x)
        return mean,std
    
    def sampling(self, mean, std):
        epsilon = torch.rand_like(std).to(self._device)
        z = mean + std * epsilon
    
    def decode(self, x):
        x = self.generation_layer(x)
        return self.decoder(x)

    def forward(self, x):
        mean, std = self.encode(x)
        z = self.sampling(mean, std)
        x_hat = self.decoder(z)
        return x_hat, mean, std
```

损失函数：
1. 重建损失: 
```
recon_loss=torch.nn.functional.mse_loss(recon,x,reduction='sum')
```
2. KL散度:

$$
KL(N(\mu, \sigma^2),N(0,1)) = \frac{1}{2}(\mu^2+\sigma^2-2log(\sigma)-1)
$$

```
kl_loss = -0.5*torch.sum(1+2*torch.log(std)-mean.pow(2)-std.pow(2))
kl_loss = torch.sum(kl_loss)
```

## VQ-VAE(Vector Quantised VAE)
VQ-VAE主要是在VAE的基础上引入了离散的、可量化的隐空间表示，有助于模型更好的理解数据中的离散结构和语义信息，同时可以避免过拟合。

### Vector Quantization技术
将连续的变量映射到一组离散的变量中，在深度学习中，VQ通常用于将连续的隐空间表示映射到一个有限的、离散的CodeBook中。

### 模型结构
VQ-VAE 与 VAE 的结构非常相似，只是中间部分不是学习概率分布，而是换成 VQ 来学习 Codebook。
1. Encoder：将输入压缩成一个中间表示
2. VQ：
    - 使用每一个中间表示与 Codebook 计算距离
    - 计算与 Codebook 中距离最小的向量的索引（Argmin）
    - 根据索引从 Codebook 中查表，获得最终量化后的表示
3. Decoder：将量化后的表示输入Decoder生成最终输出

### 模型训练
在 VQ 中使用 Argmin 来获取最小的距离，这一步是不可导的，因此也就无法将 Decoder 和 Encoder 联合训练，针对这个问题，作者添加了一个 Trick，直接将量化后表示的梯度拷贝到量化前的表示，以使其连续可导。

### 代码实现

```
def VQ(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VQ, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform(-1/self._num_embeddings, 1/self.__num_embeddings)

        self._commitment_cost = commitment_cost
    
    def forward(self, inputs):
        # BCHW to BHWC
        inputs = inputs.permute(0,2,3,1).continuous()
        input_shape = inputs.shape

        # Flatten input (DC)
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                   + torch.sum(_embedding.weight**2, dim=1)
                   - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
        
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encoding.scatter_(1, encoding_indices,1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized-inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs*torch.log(avg_probs+1e-10)))
        
        return loss, quantized.permute(0,3,1,2).continuous(), perplexity, encoding
```

### Loss函数

基于Straight-Through Estimator设计梯度：
$$
L1=||x-decoder(z+sg[z_q-z])||_2^2
$$

确保$z_q$和$z$尽量相似，实现$z_q$是z的聚类中心：
$$
L2=||z-z_q||_2^2
$$

由于编码表$z_q$是相对自由的，而z要尽力保证重构效果，所以应当尽量“让$z_q$接近$z$”而不是“让$z$接近$z_q$”，因此将损失分解为:

$$
L2=\beta||sg[z]-z_q||_2^2+\gamma||z-sg[z_q]||_2^2, \, \, \gamma = 0.25\beta
$$


### 图像生成

其实上述编码的过程，可以看作是把原图映射成了一个更小的“图像”。

映射成“小图“的过程，其实和AE的数据压缩是类似的。赋予其生成能力的，是PixelCNN。

VQ-VAE的工作过程。
1. 训练VQ-VAE的编码器和解码器，使得VQ-VAE能把图像变成「小图像」，也能把「小图像」变回图像。
2. 训练PixelCNN，让它学习怎么生成「小图像」。
3. 随机采样时，先用PixelCNN采样出「小图像」，再用VQ-VAE把「小图像」翻译成最终的生成图像。

## VQ-GAN
VQ-GAN的整体架构大致是将VQVAE的编码生成器从PixelCNN换成了Transformer（GPT-2），并且在训练过程中使用PatchGAN的判别器加入对抗损失。

### 重建模型

$$
L_{VQ} = L_{perceptual}(x,\hat{x}) + \beta||sg[z]-z_q||_2^2+\gamma||z-sg[z_q]||_2^2
$$

$$
L_{GAN} = logD(x)+log(1-D(\hat{x}))
$$


### 生成模型

1. 根据第一阶段的模型，得到Zq，设其大小为[H,W,N]。N就是codebook中向量的长度。
2. 将其平展到[H*W, N]，记为unmodified_indices。
3. 随机替换掉一部分的unmodified_indices。具体实现上，是先生成一个0-1分布的bernoulli分布的mask，以及一个随机生成的random_indices，然后通过下式来计算：
4. Transformer的学习过程即为：喂入modified_indices，重构出unmodified_indices。训练损失函数为cross-entropy。


# 参考文章

[文生图模型演进：AE、VAE、VQ-VAE、VQ-GAN、DALL-E 等 8 模型](https://mp.weixin.qq.com/s?__biz=Mzk0ODU3MjcxNA==&mid=2247485323&idx=1&sn=4408ac639f54f87c62cb64503cc2e9d9&chksm=c364c0cef41349d8f7a0c2d388b3de7bdfef049c8024b09e382e20a8e337e7c7acbca7b0a8e7&scene=178&cur_album_id=3154744884457668609#rd)

[【多模态】AE、VAE、VQVAE、VQGAN原理解读](https://zhuanlan.zhihu.com/p/657857297)