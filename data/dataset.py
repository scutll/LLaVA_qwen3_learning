from datasets import load_dataset, DownloadConfig
from transformers import AutoTokenizer, AutoProcessor
from transformers import DataCollatorForSeq2Seq
from torch.utils.data import Dataset
from PIL import Image
import torch
import io
import os

base_path = os.path.dirname(os.path.abspath(__file__))
# Tokenizer
qwen_path = os.path.join(base_path, "..", "Qwen3-0.6B")
clip_path = os.path.join(base_path, "..", "clip-vit-base-patch16")


def find_assistant_tokens(tokenizer, token_ids):
    """用于多轮对话中，找到所有assistant的起止位置"""
    result = []
    start_index = 0
    assistant_id = tokenizer("assistant", add_special_tokens=False)["input_ids"][0]
    im_end_id = tokenizer.eos_token_id

    while start_index < len(token_ids):
        try:
            first_assistant_idx = token_ids.index(assistant_id, start_index)

            try:
                corresponding_im_end_idx = token_ids.index(
                    im_end_id, first_assistant_idx
                )
                result.append((first_assistant_idx + 1, corresponding_im_end_idx))
                start_index = corresponding_im_end_idx + 1

            except ValueError:
                break

        except ValueError:
            break

    return result

# 用于再vlm_train.py训练文本生成能力和图文信息对齐能力的数据集
# image: 图片字节
# input: 指令(请概括图片内容)
# output: 对图片内容的概括
class VLMDataset(Dataset):
    def __init__(
        self,
        qwen_path,
        clip_path,
        config=None,
        split_type="train",
        val_split_ratio=0.05,
    ):
        super().__init__()
        print(f"加载 {split_type} 数据集 (tejasvaidhya/llava-cc3m-pretrain-595K)")

        # 加载原始数据 image + input + output
        raw_dataset = load_dataset(
            "tejasvaidhya/llava-cc3m-pretrain-595K",
            split="train",
            cache_dir=base_path,
            download_config=DownloadConfig(resume_download=True),
        )

        # 随机分配训练集和验证集，seed是随机数种子，用什么都行
        split_dataset = raw_dataset.train_test_split(test_size=val_split_ratio, seed=42)
        
        if split_type == "train":
            self.dataset = split_dataset["train"]
            print(f"训练集大小: {len(self.dataset)}")
        else:
            self.dataset = split_dataset["test"]
            print(f"验证集大小: {len(self.dataset)}")

        # 加载图片和文字的词化器
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_path)
        self.processor = AutoProcessor.from_pretrained(clip_path)
        self.image_pad_num = config.image_pad_num

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # 图片字节转变为rgb字节(3, 224, 224)
        image_bytes = item["image"]
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != "RGB":
            image = image.convert("RGB")

        pixel_values = self.processor(images=image, return_tensors="pt")[
            "pixel_values"
        ].squeeze(0)

        # 制作输入提示词input_ids, 用于跟图片的向量信息贴在一起
        user_prompt = item["input"]
        if "<image>" not in user_prompt:
            user_prompt += "\n<image>"

        # 提问文本
        q_text = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        ).replace("<image>", "<|image_pad|>" * self.image_pad_num)

        # 回答文本
        a_text = item["output"] + self.tokenizer.eos_token

        # 两串文本进行token化
        q_input_ids = self.tokenizer(q_text)["input_ids"]
        a_input_ids = self.tokenizer(a_text)["input_ids"]

        # 给模型的输入: input + <image>(晚点会插进去) + output
        input_ids = q_input_ids + a_input_ids
        # label: pad * len(input + <image>) + output 
        labels = [self.tokenizer.pad_token_id] * len(q_input_ids) + a_input_ids

        # 去掉input_ids的最后一个token, 也就是output的最后一个token <eos>
        input_ids = input_ids[:-1]
        # 去掉第一个pad
        labels = labels[1:]
        # 这样labels的第i个token对应input_ids的第i+1个token

        # input_ids = [101, 200, 201, 300, 301, EOS]
        # labels    = [PAD, PAD, PAD, 300, 301, EOS]

        # =>
        
        # input_ids = [101, 200, 201, 300, 301] 
        # labels    = [PAD, PAD, 300, 301, EOS]

        # 这样的原因是Decoder在输入序列长度到t的input_ids的输出是logits[t],也就是对第t+1个token的预测, 也就是labels[t],这样就对齐了

        # 在训练的过程中，每计算出logits[t]都可以计算一次loss，直到模型自己生成了[EOS], 或者达到了labels的output的最大长度就停止
        # 在推理中，模型就自己来决定什么时候该是EOS了


        # 返回input_ids,labels和图像源数据
        return {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": pixel_values,
        }



# VLM的DataCollator
class VLMDataCollator(DataCollatorForSeq2Seq):
    
    def __call__(self, features, return_tensors=None):
        # 所有pixel_values取出来堆叠成batch -> (batch_size, 3, 224, 224)
        pixel_values = torch.stack(
            [feature.pop("pixel_values") for feature in features]
        )

        # 交给父类处理input_ids和labels的padding
        # 将input_ids补padding到统一的长度，才能放进一个batch
        # 同时生成attention_mask来告诉模型哪些是valid的token哪些是padding
        batch = super().__call__(features, return_tensors)

        # 图像特征塞回去
        batch["pixel_values"] = pixel_values
        
        # 返回一个batch的数据
        return batch
        # {
        #     'input_ids', 'labels', 'attention_masks', 'pixel_values' 
        # }



# 用于LoRA微调的数据集，跟上面的差不多都是图片+问+答，不过问的会更详细或多角度，回答也会描述的更详细
# 而且问答有统一的json格式
class LoRADataset(Dataset):
    def __init__(
        self,
        qwen_path,
        clip_path,
        config=None,
        split_type="train",
        val_split_ratio=0.05,
    ):
        super().__init__()
        dataset_name = "trl-lib/llava-instruct-mix"
        print(f"加载 {split_type} 数据集 ({dataset_name})")

        raw_dataset = load_dataset(dataset_name, split="train", cache_dir=base_path)

        split_dataset = raw_dataset.train_test_split(test_size=val_split_ratio, seed=42)
        # 这里有分train和test的数据集
        if split_type == "train":
            self.dataset = split_dataset["train"]
            print(f"训练集大小: {len(self.dataset)}")
        else:
            self.dataset = split_dataset["test"]
            print(f"验证集大小: {len(self.dataset)}")
        
        # 加载Qwen和CLIP的tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_path)
        self.processor = AutoProcessor.from_pretrained(clip_path)
        self.image_pad_num = config.image_pad_num
        self.pad_token_id = self.tokenizer.pad_token_id

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # 处理图片，和VLMDataset的一样
        image = item["images"][0]
        if image.mode != "RGB":
            image = image.convert("RGB")

        pixel_values = self.processor(images=image, return_tensors="pt")[
            "pixel_values"
        ].squeeze(0)


        # 每个样本有images, prompt(多轮问答和最后一个提问), completion(回答)
        # 这一个样本和上面VLM的最大不同就是prompt包含了好几轮的关于图片的问答,然后再问一个问题, 答案在completion中, 这就是指令微调的数据集标准形式
        
        # 完全完整的几轮问答, 在第一轮问答的问题后面加上图片
        messages = item["prompt"] + item["completion"] 
        if "<image>" not in messages[0]["content"]:
            messages[0]["content"] += "\n<image>"

        # 完整的文本,并将<image>替换为49个<|image_pad|>
        full_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        ).replace("<image>", "<|image_pad|>" * self.image_pad_num)
        
        # 词元化
        full_input_ids = self.tokenizer(full_text)["input_ids"]
        
        # prompt的第一轮对话的问题中附上图片,并进行上面一样的处理
        prompt_messages = item["prompt"]
        if "<image>" not in prompt_messages[0]["content"]:
            prompt_messages[0]["content"] += "\n<image>"
         
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        ).replace("<image>", "<|image_pad|>" * self.image_pad_num)
        
        # prompt长度,后面切割用
        prompt_len=len(self.tokenizer(prompt_text)["input_ids"])
        
        # labels是completion的内容,前面用pad补齐长度
        labels=[self.pad_token_id]*prompt_len+full_input_ids[prompt_len:]
        
        # logit[t]和label[t]对齐的操作
        input_ids=full_input_ids[:-1]
        labels=labels[1:]


        # input_ids: prompt(image嵌在第一个user问题里) + completion
        # labels: padding + completion
        # pixel_values: bytes of picture
        return {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": pixel_values,
        }




if __name__ == "__main__":
    dataset = LoRADataset(qwen_path, clip_path)
    # print(random.choice(dataset))
