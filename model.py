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
        qwen_path=qwen_path,
        clip_path=clip_path,
        image_pad_num=49,
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
        self.qwen = AutoModelForCausalLM.from_pretrained(
            config.qwen_path, dtype=config.dtype
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.qwen_path)
        self.clip = AutoModel.from_pretrained(
            config.clip_path, dtype=config.dtype
        )
        self.dense1 = nn.Linear(
            self.clip.config.vision_config.hidden_size * 4, # patch16 version
            self.qwen.config.hidden_size,
            dtype=config.dtype,
        )
        self.dense2 = nn.Linear(
            self.qwen.config.hidden_size,
            self.qwen.config.hidden_size,
            dtype=config.dtype,
        )
        self.image_pad_num = config.image_pad_num
        self.pad_id=self.tokenizer.pad_token_id
        self.image_pad_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")

        for param in self.clip.parameters():
            param.requires_grad = False

        if config.qwen_frozen:
            for param in self.qwen.parameters():
                param.requires_grad = False

    def forward(self, input_ids, labels, pixel_values, attention_mask=None):
        text_embeds = self.qwen.get_input_embeddings()(input_ids)
        image_embeds = self.clip.vision_model(pixel_values).last_hidden_state[
            :, 1:, :
        ]  # 去掉cls token
        b, s, d = image_embeds.shape    # 如果使用patch16，可以解除这两行注释
        image_embeds = image_embeds.view(b, -1, 4 * d)  # (b, 49, 768 * 4) 

        image_features = self.dense2(F.silu(self.dense1(image_embeds)))
        text_embeds = text_embeds.to(image_features.dtype)

        text_embeds = self.merge_text_and_image(input_ids, text_embeds, image_features)
        qwen_outputs = self.qwen(
            inputs_embeds=text_embeds, attention_mask=attention_mask
        )
        logits = qwen_outputs.logits

        loss = None
        if labels is not None:
            loss_func = nn.CrossEntropyLoss(ignore_index=self.pad_id)
            loss = loss_func(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1).to(logits.device),
            )

        return CausalLMOutputWithPast(loss=loss, logits=logits)

    def merge_text_and_image(self, input_ids, text_embed, image_embed):
        batch_indices, image_indices = torch.where(input_ids == self.image_pad_id)
        text_embed[batch_indices, image_indices] = image_embed.view(
            -1, image_embed.shape[-1]
        )
        return text_embed


if __name__ == "__main__":
    config = VLMConfig()
    vision_language_model = VLM(config)
    params = sum(
        p.numel() for p in vision_language_model.parameters() if p.requires_grad
    )
    print(f"Trainable params: {params/1e6}M")
