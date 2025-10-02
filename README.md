# LLaVM: Large Language and Vision Assistant多模态多轮对话模型

​	本项目为对LLaVM的大致解构和分析，包含数据集、模型、训练、微调等各方面经验，旨在掌握当前多模态大模型微调方                                                                                                 法以及探索扩展到中文语言模型的开发。

​	项目使用HuggingFace的数据集、模型，并使用LoRA方法进行微调。

- 大语言模型: 	https://huggingface.co/Qwen/Qwen3-0.6B
- 视觉embedding主干: https://huggingface.co/openai/clip-vit-base-patch32
- 数据集：
  - 视觉-文本对齐：https://huggingface.co/datasets/tejasvaidhya/llava-cc3m-pretrain-595K
    - 刚开始可能不是很理解对齐的意义，其实就是要训练线性层使得CLIP输出的图片向量信息和经过qwen的embedding层的向量**信息能够对齐**
  - 微调：https://huggingface.co/datasets/trl-lib/llava-instruct-mix

环境：

- python 3.11.13(Conda)
- cuda 12.4或更高
- torch: 2.6.0+cu124
- transformers: 4.56.1 



## 模型

​		模型使用预训练的Qwen3-0.6B的文本生成模型以及clip-patch32进行图像识别

### Qwen预训练模型

- Qwen是一个Decoder-only模型，在训练中它接收经过embedding处理后的文字以及图片作为输出，输出对每个词的预测向量即logits, 这是对每个embed数据并行计算后得到的多个vocab_size大小的向量

  Qwen在训练过程中要经过这些操作：

  - 对batch_size个长度为seq_length**句子进行tokenize**, 即按照vocab将input_text映射成input_ids
    - (b, L, 1) -> (b, L, 1) 这里是将句子转化成tensor
  - 然后就要**输入到embedding层进行向量化**，每个token经过embedding层都会转化成一个长为H(最后全连接层的size)的向量，这个向量包含了词语的含义
    - (b, L, 1) -> (b, L, H)
  - 这时还要将input_ids中代表**image_pad**的向量换成图片的向量, 图片也是被处理成(batch_size, num_patch, H)形状的向量，然后按照image_pad的顺序将图片信息填入text_embed中，此时形状没有被改变。这就完成了**文本和图片信息的融合**
  - 然后就可以将融合后的text_embed丢进Decoder里面获取logits，logits是对每个token中所有词可能性的预测
  - 同时别忘了attention_mask, 这是一个用来屏蔽padding_token的掩码
    - (b, L, H) -> (b, L, vocab_size)
  - 若要计算loss，可以将logits丢进softmax层和交叉熵层进行计算

### CLIP Contrastive Language–Image Pretraining 跨模态模型

- CLIP的作用是将图像当作文字来识别，通过向量化将**图片信息同化为跟文字在qwen的embedding层处理后的一样的一堆H长度的向量**，这样就可以将文字信息和图片信息融合起来，一起交给大模型进行信息提取和学习推理
- 首先要介绍的是patch，在CLIP中，我们将**一个图片分成32x32或者16x16大小的一个个小块**，称为一个patch，这样我们就可以将图片当作一个个词语来进行理解。一张图片的大小为224x224，若分割大小为32x32，那么就可以分割出(224/32)**2 = 49个patch，因此patch32的块数是49.
- 将图片分成patch以后，我们就可以将它丢进Vision Model中进行一些操作(具体啥你别管)来提取图片中的信息，最后也是通过线性层(应该)变成一个个num_hidden长度的向量组，具体形状就是(batch_size, num_patch, num_hidden)，但为**了跟文字信息进行融合，我们就需要将num_hidden大小的向量转化成H的大小**，很容易就可以想到我们的老朋友线性层，在模型中也是利用了两个线性模块进行信息继续提取，最终转化成形状为(b, p, H)的tensor，这样就可以跟文本信息进行融合了
- 所以简单来说，CLIP作为一个图文对齐模型，使得图片和文字可以映射到同一个空间，相当于把图片转化成跟文字相近的语义向量(也就是说**意思差不多的图片和文字应该会有差不多的向量表现**)，这样就可以一起作为信息输入到Decoder进行分析

### merge_text_and_image

- 这是一个将文字信息和图片信息融合到一起的函数，embed层中的每个向量长度是H，也就是qwen的embedding层输出的向量长度，我们要定位input_ids中的image_pad，然后依次用batch_size * num_patch个图片向量替换掉image_pad，制作成text_embed，包含**图片和文本的混合信息**

### VLM Vision Language Model

- **VLM就是Qwen和CLIP结合的模型，其主干依旧是Qwen的Decoder**，但将CLIP的图像识别功能也集成到一起，因此有了理解文本和图片的功能。
- VLM的输入：
  - input_ids：经过token化后的文本，每一个id表示一个词语
  - labels： 文本推断的正确答案
  - pixel_values：图片像素信息
  - attention_mask：注意力掩蔽用的mask，或者是用于过滤pad用的mask
- VLM的输出：loss和logits(对每句话中每个词的词典向量概率推测)
- config信息：
  - qwen_path, clip_path，文本和图像识别模型的导入地址
  - image_pad_num：image_pad的填充数，若是patch32就是49个
  - dtype：数据形式
  - qwen_frozen：进行微调时使用LoRA方法，并不改变原始权重，因此冻结权重
- 模型训练forward过程图解
  - 最后的输出就是loss和logits
- 通过详细分析数据集，终于理解了这里训练得到logits的逻辑，本质上是**并行计算**。我们先简单看下数据：
  - input_ids： 一段问题 + 嵌入图片 + 回答内容
  - labels：跟input_ids问题和图片长度相同的paddings + 回答内容
  - 然后input_ids和labels分别去掉最后一个token(<eos>)和第一个token(<pad>)来使得**logits[t]和labels[t]对齐，这样logits[t]就对应是labels[t]的预测结果**

#### 并行计算：input_ids和labels这个形状的原因和模型的训练过程

- 

  - 然后我们来看计算，我们一次性把logits全计算出来，**每个logits[t]代表着将input_ids[0: t]经过embedding层并输入qwen得到的对第t+1个token的预测结果**，而这个结果可以跟labels[t]比对进行loss的计算。在output内容之前的是问题和图片的内容，我们可以**省略掉不加入loss的计算**(attention_mask)。而**每次计算logits我们都是用正确答案output的token序列，因此每个logit的计算并不影响另一个**，所以我们可以并行地计算所以logits，然后依次与labels进行loss计算(如图)。

  ![1759131550878](C:\Users\mxl_scut\AppData\Roaming\Typora\typora-user-images\1759131550878.png)

  - 如果不并行看的话，就相当于从input_ids[0: t] (t=len_input_and_image)开始，计算一个logits[t]，然后跟label[t]比对计算loss，然后又放入input_ids[0:t+1]计算logits[t+1]，跟labels[t+1]比对...直到计算完input_ids的整条序列，而这一个个序列的计算是互不干扰的。**本质上还是给一个序列然后预测下一个词，但这里妙就妙在可以并行计算一次性算出来所有logits**

![img](file:///C:/Users/mxl_scut/Downloads/whiteboard_exported_image.png)

------

## Inference 推断

​		推断就是加载微调好的模型，并开启接收输入和产生输出的能力

​		在inference.py中，我们实现了一个generate_response函数，这与之前做过的ai古诗生成的目的相同，都是去**使得模型能按照需要的格式去输出**，但这一次要更加简单，因为我们只需要保证模型的输出长度小于某个值和让它在生成出一个<eos>的token的时候停止即可

​		在文件中，我们实现了一个单轮对话的功能，大概就是输入文本，图片，然后大模型输出回答而已

### 前置工作

​		在输入之前，需要准备我们的多模态模型：

- 加载Qwen和CLIP的编码器

- 设置GPU

- 创建VLM并导入训练好的模型权重

  这样就可以开始对话推理模式了

### 生成

- 输入text文本和图片后，首先要做的肯定是**通过embedding层将文本和图片转化成Qwen能看懂的信息**
  - 对于文本：我们在进行tokenization之前还要**将一个<image>占位符插入文本，然后将它替换为49个<|image_pad|>, 这样就得到了template_prompt**，然后将文本插入**一个和训练格式一致的对话模板**（如下）,还要进行tokenizer专属的对话模板处理: 最终得到一个**formatted_prompt**，才可以进行token化得到**input_ids**

```python
conversation = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": template_prompt}
            ]
formatted_prompt = tokenizer.apply_chat_template(
                conversation, tokenize=False, 
                add_generation_prompt=True,
                # 在最后加上助手的开头提示（比如 \nAssistant:），让模型知道下一步要输出的是“助手的回答”
                enable_thinking=False
                # 如果为 True，会在 prompt 里插入思维链标记（类似 <think>）
            )
```

- - 对于图片： 需要读取image并获得**pixel_values(batch_size=1, 3, 224, 224)**
- 获得input_ids和pixel_values我们就可以输入到qwen模型里面进行token猜测了

#### Decoder 的文本序列生成

- 作为一个Decoder-only模型，Qwen在推断模式时接受embed_text**(此处省略input_ids和pixel_values的同步信息和合并过程) **作为输入，生成对下一个token的猜测logits => ** (batch_size=1, vocab_size)

  - 在这里，我们挑选概率最大的token作为输出，也就是vocab_size维度中的最大值
  - **思考：如果我们不选用最大可能性的token而是在前k个可能的token中随机选择，模型输出序列的多样性会不会更加丰富**

- 回归正题，选择到输出token后，我们要做的就是**将这个token放到input_ids后面，然后再用新的input_ids进行下一个token的预测**，直到达到最长序列生成长度或者是预测的token是<eos>就结束。这也就是Decoder-only模型的文本序列生成方式

- 当模型停止输出token，我们就只需要将原始input_ids以后的tokens拿出来然后通过tokenizer的decode功能映射回文本序列就可以得到大模型对我们问题的输出了

  ------

## VLM_Train 多模态模型训练（stage1)

- 训练直接使用的trainApi，没什么看点，无非就是前向传播和梯度下降，这里主要介绍**训练过程中的一些超参数和一些细节**
- 首先是一个问题，这是一个多模态大模型，但在vlm_train.py中，并没有找到跟图片相关的东西，也就是在**train的过程中似乎并没有将图片放入qwen中进行训练**，可在inference中明显是要放入图片进行理解的。
  - 这就凸显了多模态中处理图片的逻辑：把图片当作文本一样去理解。**qwen并不负责看图片，它只是接受CLIP提取图片特征信息后理解到的信息，并把它当作文本一样去理解**。简单来说，就是CLIP来负责理解图片，并告诉qwen这个图片里有什么信息。
  - 这里就引出了一个很重要的操作：**信息对齐**，因为CLIP和qwen并不认识，他们互相不能理解对方说的语言，也就是**CLIP产生的图片向量信息并不能被qwen直接理解**，因此我们加入两个dense层。一是将CLIP的图片向量维度和qwen文本的embedding层的向量维度对齐，二就是**要使得CLIP产生的图片信息向量对齐到qwen能理解的文本信息向量空间中**(也就是进行翻译)。当信息对齐了，qwen就可以理解CLIP对图片的描述了
  - 因此，除了对qwen文本理解能力的微调训练，对CLIP特征提取能力的微调训练，我们还要训练的一项能力就是**从图片信息向量对齐到embedding文本信息向量，说白了就是那俩线性层**
  - 实际上观察VLMDataset的代码，里面包含了图片和文本描述的数据集，所以**在vlm_train中，我们训练的其实就是文本生成能力和图文信息对齐能力**

------

- 然后就是超参数，这里只挑一些之前没怎么见过的或者比较重要的
  - warmup_ratio=0.05 **前5%的数据学习率从0开始线性增长直到lr**，因为一开始学习率过大可能导致loss发散或者训练不稳定
  - lr_scheduler_type='cosine' 使用cosine作为**学习率的调度衰减策略**
  - weight_decay权重衰减率
  - bf16=True, fp16=False，启用**bfloat16数据类型**而不是float16，bfloat16使用和float32一样的指数位数(8)，缩小尾数位数(7)，这样可以牺牲一些精准度来换取更大的数值范围

------

## 数据集 Dataset

​		在讨论微调之前，感觉还是有必要先看一下数据集长什么样。。文件里最主要的就是VLMDataset和

LoRADataset两个数据集，还有一个VLMDataCollator

------

### VLMDataset

- 选用用于训练llava的一个图片加文本数据集，有三部分数据：

  - 图片Image：字节格式
  - input：一段对模型的提问，大致内容就是概括图片的内容
  - output：对图片的内容描述

- VLMDataset的工作：

  - 将图片变成(3, 224, 224)的RGB三通道形状数据pixel_values
  - 制作提问文本和回答文本
    - **q_text**：json格式的文本，每一句包含键<role>和<content>，包含system提示词和user问题，这里我们将input后面加上<|image_pad|>之后作为user_prompt放入q_text的role为user的content里面
    - **a_text**：问题的回答output后面加上句子结束符号<eos>
    - 然后对a_text和q_text进行token化得到a_input_ids和q_input_ids
  - 这时候就可以制作input_ids和labels了：
    - **input_ids=q_input_ids+a_input_ids**，是给模型进行训练的序列
    - **labels=paddings + a_input_ids**， paddings数量和q_input_ids的长度一样
    - 然后我们还要分别**将input_ids的最后一个token<eos>和labels的第一个token<pad>删除掉(shift操作)**，这样logits[t]就可以对应laebls[t]，详细原因看VLMmodel的forward并行计算logits的逻辑

  ```python
  #举例说明shift操作，前三个是q_input_ids, 后三个是a_input_ids
  input_ids = [101, 200, 201, 300, 301, EOS]
   labels    = [PAD, PAD, PAD, 300, 301, EOS]
  
  =>
          
   input_ids = [101, 200, 201, 300, 301]    #去除EOS
   labels    = [PAD, PAD, 300, 301, EOS]    #去除PAD
  ```

  

  - 最后我们的返回就是**input_ids, labels和pixel_values**

------

### LoRADataset

- 用于进行LoRA微调以及训练qwen模型进行多轮对话能力的数据集
- 有三部分数据：
  - **images**：图片PIL格式，需要先转化成RGB格式
  - **prompt**：一串user和assistant之间关于图片内容的问答以及最后一个对图片内容的提问(json格式)
  - **completion**：assistant对prompt最后的问题的回答
- 这一个数据集的处理和上面VLMDataset的处理方式一样，返回的一样也是input_ids, labels和pixel_values。这个**数据集唯一的不同是在input_ids上面会有更多的内容，也就是在提出问题之前有多轮关于这一个图片的提问和回答对，因此是专门用来微调强化训练模型的多轮对话能力的**，具体LoRA微调了哪一些层的参数，就要在peft_train.py上看了

------

## LoRA微调（stage2）

- 使用LoRA对模型进行**文本生成能力的微调以及强化多轮对话能力**，使用LoRA的一个好处就是完全不用动stage1训练完的qwen模型权重参数，而是在要进行微调的一些层上添加**LoRA adapter层**来充当微调矩阵，也就是ΔW

- 简单来说，**LoRA并不修改原始权重W**，而是在原始权重基础上加上一个ΔW再跟X进行运算，即
  $$
  Y = (W + \Delta W)X  \ \ \ \ \ \ \ 或 \ \ \ \ \ \ \ Y=(W + AB)X
  $$

- 通过**把ΔW分解成AB两个低秩矩阵来大大减少参数量**

- 为了缓解AB更新太小可能学不到很多东西，我们使用缩放系数α，这样ΔW就等于 α/rank(AB)，一般来说，rank取8，也就是AB的形状分别是hiddenx8, 8xhidden，然后α取16

- 说了这么多，**实际上就是在要训练的参数层(冻结)的基础上每个参数加一个微调的偏移量，而这些偏移量可以通过分解成更低秩的几个矩阵来减少参数量**，在此基础上，这个偏移线性层的训练跟一般线性层的训练都是大差不差的，既可以添加bias偏置，也可以进行dropout操作，也可以进行weight_decay，也要设置学习率。。。那我们再来看下LoRA独有的一些重要配置参数：

  - **r：** 要分解到的秩，比如AB分别为 hidden x r, r x hidden
  - **lora_alpha：** LoRA缩放系数
  - **target_modules：** 要插入LoRA adapter进行微调的层，比如Q(q_proj)、K(k_proj)、V(v_proj)、Output(o_proj)
  - **task_type：** 不同类型模型使用的任务类型，在我们这个Decoder-only自回归模型中，我们用的是**CAUSAL_LM**，即每次生成一个token

- LoRA的配置方式：peft库提供了十分简单的配置LoRA的方式，只需配置LoRAConfig，然后用get_peft_model为satge1训练过的qwen模型注入LoRA adapter层即可进行训练

------

## 微调后的多轮对话Inference

​		经过LoRA微调后的大模型已经具备进行多轮对话的能力，对比基础版的Inference，其实并没有本质的区别，只是模型变成了嵌入LoRA adapter层的qwen模型，然后还有维护一个conversation来记录多轮对话的记录，**文字生成的基本逻辑还是每次生成一个token**

​	跟基础Inference的内容就不作过多介绍了，我们来看一下不同的一些地方：

- 首先是模型的加载，依然使用peft库的**get_peft_model**将LoRA adapter层嵌入模型，然后再使用**load_state_dict**来将权重加载到模型和adapter层中，通常我们还有进行一步**merge_and_unload来将LoRA的低秩矩阵合并到base权重中，然后卸载LoRA的所有微调层**，这样就可以大大加速推理速度和减少内存占用（缺点自然是改变了权重而且以后想要重新微调需要重新添加adapter层）
- 在加载模型后，我们才进行词化器的加载，这样是为了缓解模型加载时的内存负担。。
- 然后很重要的一步就是创建一个**conversation**，这表示的是模型跟用户多轮的对话记录，一开始只有一个role为<system>的记录(应该是作为system prompt来用)，**后面每次提问和产生回答都会将这些文字按格式添加进conversation中**，以此来维护一个完整多轮对话的历史。
- 后面图片和文字的处理过程就和基本的推断程序没什么差别了，要注意的就是在**formatted_prompt进行格式化的时候传入的是conversation**，而不只是当前的一个问题，**即每次对话都要将前面的对话历史全部告诉大模型，这样才能使得模型拥有“记忆”**。
- 最后生成回答，也是照样把回答插入conversation，这样一轮对话就完成了，一直循环就实现了多轮对话的功能。

------

## 训练实践

​		说理论说很多了，但是实际训练起来也是不轻松，首先就是autodl连接huggingface难，这里挂代理就好多了。具体的模型和数据集下载可以看**load_hf.py**这个文件一次性把需要的都下载好了。

​		然后就是一个很棘手的问题，总是报ValueError错，说我用了错误的Config，但看Qwen3-0.6B里面的config.json是没问题的，后来找到原因是系统要直接在Qwen3-0.6B文件夹里面找config.json和模型等文件。。**因此要将qwen_path设置成config.json的所在文件夹才能运行，用model.py测试一下就ok**

### 训练成效

- 总训练loss从5.6降到4.6然后到3.9，但实际上后面差不多一万个it都是维持在3.8到3.9附近，已经是比较拟合了。
- 在inference.py中使用能感受到，**CLIP能够一定程度上理解图片内容，文字对齐能力也基本可以了**。但问题是无论用户的文本是什么(即使不是提问)，模型的生成也是对图片的一个描述，而且同一个问题或者差不多的问题带来的答案是很相近的，甚至词语都没什么差别。。个人觉得是因为数据集中input的内容都是清一色的对图片提问的，并且**图片通常占据高达49个信息位，而文本input一般也没有这么长，导致在训练中通常是文本信息作为主导因此在推断中文本信息对生成token的影响力就过大了**
- 微调环节，训练也不快，三分之一的数据集大小但训练时间跟stage1差不多，loss从一开始的1.6作用到0.9，并且**一半epoches之后的提升也是基本没有了**。
- 然后是效果，images文件夹中有两张动漫截图和一张相机实拍图，测试发现**模型对真实世界场景的图片理解能力要明显优于动漫截图**，许多细节能够理解的更准确。原因可能是数据集里面比较少动漫类型的图片，但总的来说，CLIP对一些细节也是能够看到的，图文对齐能力也是有的。