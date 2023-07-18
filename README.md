# Neural Collaborative Filtering (NCF)论文学习笔记及代码实现
​
一、研究背景
       在推荐系统中使用神经网络进行协同过滤。传统的协同过滤方法，如矩阵分解，已经被广泛应用于推荐系统中。然而，它们通常依赖于线性模型，可能无法捕捉用户和物品之间复杂的非线性关系。本文旨在探索深度学习和神经网络在改善协同过滤性能方面的潜力。

二、相关知识
1、GFM（广义矩阵分解）模型：
![image](https://github.com/Xian-Gang/NCF/assets/129944803/c17975de-4eba-49a4-be1d-4b9b95f1b4fb)

 ​​​​

从论文中的解释看，user和item都通过one-hot编码得到稀疏向量，然后通过一个embedding层映射为user vector和item vector。这样就获得了user和item的隐向量，就可以使用哈达马积(Element-wise Produce)来交互user和item向量了。

2、One-hot编码
One-hot编码是一种常见的特征编码技术，用于将离散的分类特征转换为机器学习模型能够理解的数字表示。例如，如果我们有一个特征"颜色"，可能的取值有"红色"、“蓝色"和"绿色”。使用独热编码，我们可以将"红色"表示为[1, 0, 0]，"蓝色"表示为[0, 1, 0]，"绿色"表示为[0, 0, 1]。

优点：独热编码消除了分类特征之间的大小关系，避免了模型在处理这些特征时引入任意的顺序或数据偏移。同时，独热编码还允许模型更容易地识别和理解特征之间的不同，从而提高了模型的准确性。
缺点：独热编码会增加特征的维度，在某些情况下可能导致数据稀疏性增加。在使用时，需要考虑特征的取值数量和数据集的规模对内存和计算资源的要求。
3、embedding层
嵌入层（embedding）是一种将高维离散数据映射到低维连续向量空间的技术。比如，在推荐系统中，将用户和物品表示为嵌入向量，可以更好地捕捉用户和物品之间的关联性。

优点：是能够减少特征的维度，去除离散数据的无序性，并且将相关的实体映射到相似的向量空间位置。这些嵌入向量可以作为机器学习模型的输入，从而提高模型的表现和泛化能力（指机器学习模型在未见过的数据上的表现能力）。

4、哈达马积
哈达马积是一种矩阵运算，也被称为元素对应相乘，表示为A ⊙ B，
A=[[12],[3,4]]
B=[[5,6],[7,8]]
A ⊙ B = [[15, 26], [37, 48]] = [[5, 12], [21, 32]]
![image](https://github.com/Xian-Gang/NCF/assets/129944803/0d4ea86b-dfd9-49b0-ad6f-1341eb7f4da0)



5、MLP（多层感知机）模型：
![image](https://github.com/Xian-Gang/NCF/assets/129944803/5863c4e5-9c3e-48e7-b14e-2da67d1e2908)



Output Layer：最终输出只有一个值，离1越近表示user可能越喜欢这个item。与y ^ u i ​ 进行比较后，将损失进行反向传播，优化整个模型。 其Embedding层和InputLayer和GMF一样。Neural CF Layers使用的激活函数为ReLU。
![image](https://github.com/Xian-Gang/NCF/assets/129944803/b1243784-627c-4a89-9600-335185daeffb)



6、激活函数ReLU
使用的激活函数ReLU函数：对于大于零的输入，输出等于输入；对于小于等于零的输入，输出为零。ReLU函数通常具有更好的计算效率和收敛性。

7、NeuMF模型：
 
![image](https://github.com/Xian-Gang/NCF/assets/129944803/7e4ece16-5dc9-4277-a4cd-ac68cc5703a2)

 对分开的两部分进行融合。如下式所示：

![image](https://github.com/Xian-Gang/NCF/assets/129944803/eee943ce-725d-48ed-bf27-63ddab4540f8)


常用的平方差损失函数如下（这里详细可看论文）：
![image](https://github.com/Xian-Gang/NCF/assets/129944803/6d5df90d-38ce-4aef-ac24-47a4a4f5bcf4)

![image](https://github.com/Xian-Gang/NCF/assets/129944803/57c25618-a718-4a12-b81d-d5a76b720d7e)

 对似然函数取对数后取反，可以得到最小化的目标函数如下：
![image](https://github.com/Xian-Gang/NCF/assets/129944803/b3dc5459-1205-4b5c-b5ad-772089e03138)



三、研究方法
1、留白评估法
留白评估法是一种评估人工智能系统或算法的方法。它的主要目的是在没有先验知识或预设条件下，评估系统在面对新情境或未知数据时的表现和能力。

基本思想：是将训练好的模型或系统应用于新的测试集或数据，并观察其在未知情境下的行为和输出。与传统的评估方法不同，留白评估法强调系统在面对新情境时的鲁棒性和泛化能力。

2、数据集划分
数据集的划分采用负采率的方式：是指在训练二分类模型时，为了平衡正向样本和负向样本的比例，从负向样本中采样的比例。

3、数据集介绍
(1）MovieLens 1M，是MovieLens数据集中的一个版本，包含了100多万条用户对电影的评分数据。包括：用户评分数据，电影信息和用户信息。

(2) Pinterest是一个全球知名的图片分享和发现平台，用户可以通过Pinterest来收集、组织和分享自己喜欢的图片和视觉内容。

 四、实现结果（论文详细结果可看原文）
1、Top-K项目推荐
Top-K项目推荐是一种常见的推荐系统方法，它旨在根据用户的兴趣和偏好，为用户推荐最适合他们的前K个项目或物品。

![image](https://github.com/Xian-Gang/NCF/assets/129944803/b0a345d4-ae22-47f0-a8ba-7ed468ab5888)

![image](https://github.com/Xian-Gang/NCF/assets/129944803/d529d6ae-a50e-41ec-b4b8-10b25ff8fd25)
![image](https://github.com/Xian-Gang/NCF/assets/129944803/62ed30ae-1224-4196-83df-8a2be15e46a5)

 NCF方法的训练损失和推荐性能与MovieLens的迭代次数有关（系数为8）
![image](https://github.com/Xian-Gang/NCF/assets/129944803/13fec0f6-d578-41d6-9d0b-881e99e58443)

 

NCF方法的性能与每个阳性实例的阴性样本数量有关(factors=16)。图中还显示了BPR的性能，它只对一个负面实例进行采样，与一个正面实例配对学习。具有较大采样率的BPR。这表明点式对数损失比成对的BPR损失更有优势。对于这两个数据集，最佳采样率大约是3到6。在Pinterest上，我们发现当采样率大于7时，NCF方法的性能开始下降。它显示，过于积极地设置采样率可能会对性能产生不利影响。

不同层的MLP的HR上和NDCG下    K=10 
![image](https://github.com/Xian-Gang/NCF/assets/129944803/4f3314fe-ccc0-440d-a38f-4d2861ed76b7)

![image](https://github.com/Xian-Gang/NCF/assets/129944803/470d4cc9-1bb2-4fd1-948c-8dccfe7ba6b4)

 五、本人代码运行结果
 1、GMF
 ![image](https://github.com/Xian-Gang/NCF/assets/129944803/84377a09-fabb-4441-96e4-0e545b37f13e)

2、MLP
![image](https://github.com/Xian-Gang/NCF/assets/129944803/03285c29-441d-4ba4-8b49-f22f843100ec)

3、NeuMF
![image](https://github.com/Xian-Gang/NCF/assets/129944803/03eca3b7-bccc-4818-8dd9-2dae122733f4)

同时会产生对应的预训练文件：
![image](https://github.com/Xian-Gang/NCF/assets/129944803/d3512309-c791-4073-bddf-9d8bc751654f)


 

 

 

​
