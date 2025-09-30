import torch
from torch import nn
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PretrainedConfig,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
import os

base_path = os.path.dirname(os.path.abspath(__file__))
qwen_path = os.path.join(base_path, "Qwen3-0.6B")
clip_path = os.path.join(base_path, "clip-vit-base-patch16")


class VLMConfig(PretrainedConfig):
    model_type = "vlm"

    def __init__(
        self,
        # qwen语言模型
        qwen_path=qwen_path,
        # clip视觉模型
        clip_path=clip_path,
        # 图像patch的占位符数量
        # 图像patch就是将整张图像裁切成一小块一小块的区域，然后将小块当作“词”输入模型
        # patch32模式即每个patch32x32, 224x224被切成49个patch
        image_pad_num=49,
        # 冻结qwen参数
        qwen_frozen=True,
        dtype=torch.bfloat16,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.qwen_path = qwen_path
        self.clip_path = clip_path
        self.image_pad_num = image_pad_num  # patch32 version: (224/32)**2 = 49
        self.qwen_frozen = qwen_frozen
        self.dtype = dtype


class VLM(PreTrainedModel):
    def __init__(self, config: VLMConfig):
        super().__init__(config)
        # 加载预训练的qwen和视觉模型
        # Qwen作为语言生成backbone
        # CLIP作为一个图文对齐模型，使得图片和文字映射到同一个空间，可以说相当于将图片转化成文字(相近的语义向量)然后输入到Qwen进行分析
        self.qwen = AutoModelForCausalLM.from_pretrained(
            config.qwen_path, dtype=config.dtype
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.qwen_path)
        self.clip = AutoModel.from_pretrained(
            config.clip_path, dtype=config.dtype
        )
        # 线性层
        # 作用是将clip读取图片信息生成的向量通过线性层转化成qwen可读的信息
        self.dense1 = nn.Linear(
            # 在clip-patch16中, hidden_size = 768, 后面要将4个patch拼在一起，所以是*4
            self.clip.config.vision_config.hidden_size * 4, # patch16 version
            # Qwen的embedding维度
            self.qwen.config.hidden_size,
            dtype=config.dtype,
        )
        # dense1之后的dense2是不改变形状而进行特征继续细化
        self.dense2 = nn.Linear(
            self.qwen.config.hidden_size,
            self.qwen.config.hidden_size,
            dtype=config.dtype,
        )
        
        self.image_pad_num = config.image_pad_num
        
        # 文本padding
        self.pad_id=self.tokenizer.pad_token_id
        # 图像专用占位符，在 forward 时会被替换为图像投影后的 embedding
        self.image_pad_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")

        # 冻结参数
        for param in self.clip.parameters():
            param.requires_grad = False

        if config.qwen_frozen:
            for param in self.qwen.parameters():
                param.requires_grad = False

    def forward(self, input_ids, labels, pixel_values, attention_mask=None):
        
        # 输入input_ids转成embedding
        text_embeds = self.qwen.get_input_embeddings()(input_ids)
        # 图像处理
        # 去掉[CLS]
        image_embeds = self.clip.vision_model(pixel_values).last_hidden_state[
            :, 1:, :
        ]  # 去掉cls token
        
        # (batch_size, patch数, 每个patch被编码成的向量维度)
        b, s, d = image_embeds.shape    # 如果使用patch16，可以解除这两行注释
        image_embeds = image_embeds.view(b, -1, 4 * d)  # (b, 49, 768 * 4) 

        # 进入线性层提取特征
        image_features = self.dense2(F.silu(self.dense1(image_embeds)))
        text_embeds = text_embeds.to(image_features.dtype)  #数据类型一致
        # 合并文本信息和图像信息
        text_embeds = self.merge_text_and_image(input_ids, text_embeds, image_features)
        
        qwen_outputs = self.qwen(
            inputs_embeds=text_embeds, attention_mask=attention_mask
        )
        # logits就是模型的输出(对每一个词的预测向量) (batch_size, seq_len, vocab_size)
        logits = qwen_outputs.logits
        # 这里看懂了, 看到数据集中的内容: 
        # inputs_ids是input+image+output
        # 而labels是 padding+output
        # 经过shift后, 这里通过embedding层后丢进qwen得到logits, 其实是并行计算的结果
        # 比如logits[t], 这一个可以跟labels[t]比对计算得到loss,这里的logits[t]相当于将input_ids[t]以及之前的token序列放入qwen进行训练,得到第t+1个token的预测结果,
        # 然后logits[t+1]就是将input_ids[t+1]经过qwen的结果,这些过程可以并行计算得到,所以这里的实际训练是从input_ids的output内容开始, 依次(每一个序列都比原来的序列长1)计算logits, 这些logits就可以一个个跟labels里面的词比对,索引是一对一的.而logits的计算可以并行计算得到

        loss = None
        if labels is not None:
            loss_func = nn.CrossEntropyLoss(ignore_index=self.pad_id)
            loss = loss_func(
                # 展平为(b*l, vocab_size) 经过softmax层再进行交叉熵计算
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1).to(logits.device),
            )

        # 返回loss和logits
        return CausalLMOutputWithPast(loss=loss, logits=logits)

    def merge_text_and_image(self, input_ids, text_embed, image_embed):
        # text_embed -> (b, L, H)
        # image_embed -> (b, patch数, H) 经过dense1的转化,两个H是相同的
        
        # 这两个分别是每个匹配项对应的batch行索引和列索引,长度均为k(<|image_pad|>的个数)
        # 也就是这俩是用来显示哪一些格子是image_pad
        # batch_indices说明了有哪些行有([0,0,1,1])表示0/1行各有两个
        # image_indices说明对应行的那一列是image_pad
        # 这两个组合就得到image_pad的在input_ids(batch_size, seq_len)的具体位置
        batch_indices, image_indices = torch.where(input_ids == self.image_pad_id)
        
        # 这里就是将文字序列中image_pad的信息替换成一整个图片的信息
        # 先将image_embed变形(batch_size,num_patches,emb_num_hidden) ->(b*n, h)
        # 每一个text_embed塞进一个patch, 也就是长为H的向量,也就是49个连续的image_pad就塞入了一整张图片的信息
        text_embed[batch_indices, image_indices] = image_embed.view(
            -1, image_embed.shape[-1]
        )
        
        # 这里就是文本信息和图片信息合在一起后的信息
        return text_embed


if __name__ == "__main__":
    config = VLMConfig() 
    vision_language_model = VLM(config)
    params = sum(
        p.numel() for p in vision_language_model.parameters() if p.requires_grad
    )
    print(f"Trainable params: {params/1e6}M")
