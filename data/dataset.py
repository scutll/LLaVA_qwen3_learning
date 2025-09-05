from datasets import load_dataset, DownloadConfig
from transformers import AutoTokenizer, AutoProcessor
from transformers import DataCollatorForSeq2Seq
from torch.utils.data import Dataset
from PIL import Image
import torch
import io
import os

base_path = os.path.dirname(os.path.abspath(__file__))
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

        raw_dataset = load_dataset(
            "tejasvaidhya/llava-cc3m-pretrain-595K",
            split="train",
            cache_dir=base_path,
            download_config=DownloadConfig(resume_download=True),
        )

        split_dataset = raw_dataset.train_test_split(test_size=val_split_ratio, seed=42)
        if split_type == "train":
            self.dataset = split_dataset["train"]
            print(f"训练集大小: {len(self.dataset)}")
        else:
            self.dataset = split_dataset["test"]
            print(f"验证集大小: {len(self.dataset)}")

        self.tokenizer = AutoTokenizer.from_pretrained(qwen_path)
        self.processor = AutoProcessor.from_pretrained(clip_path)
        self.image_pad_num = config.image_pad_num

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        image_bytes = item["image"]
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != "RGB":
            image = image.convert("RGB")

        pixel_values = self.processor(images=image, return_tensors="pt")[
            "pixel_values"
        ].squeeze(0)

        user_prompt = item["input"]
        if "<image>" not in user_prompt:
            user_prompt += "\n<image>"

        q_text = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        ).replace("<image>", "<|image_pad|>" * self.image_pad_num)

        a_text = item["output"] + self.tokenizer.eos_token

        q_input_ids = self.tokenizer(q_text)["input_ids"]
        a_input_ids = self.tokenizer(a_text)["input_ids"]

        input_ids = q_input_ids + a_input_ids
        labels = [self.tokenizer.pad_token_id] * len(q_input_ids) + a_input_ids

        input_ids = input_ids[:-1]
        labels = labels[1:]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": pixel_values,
        }


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
        if split_type == "train":
            self.dataset = split_dataset["train"]
            print(f"训练集大小: {len(self.dataset)}")
        else:
            self.dataset = split_dataset["test"]
            print(f"验证集大小: {len(self.dataset)}")

        self.tokenizer = AutoTokenizer.from_pretrained(qwen_path)
        self.processor = AutoProcessor.from_pretrained(clip_path)
        self.image_pad_num = config.image_pad_num
        self.pad_token_id = self.tokenizer.pad_token_id

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"][0]
        if image.mode != "RGB":
            image = image.convert("RGB")

        pixel_values = self.processor(images=image, return_tensors="pt")[
            "pixel_values"
        ].squeeze(0)

        messages = item["prompt"] + item["completion"]
        if "<image>" not in messages[0]["content"]:
            messages[0]["content"] += "\n<image>"

        full_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        ).replace("<image>", "<|image_pad|>" * self.image_pad_num)
        
        full_input_ids = self.tokenizer(full_text)["input_ids"]
        
        prompt_messages = item["prompt"]
        if "<image>" not in prompt_messages[0]["content"]:
            prompt_messages[0]["content"] += "\n<image>"
         
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        ).replace("<image>", "<|image_pad|>" * self.image_pad_num)
        
        prompt_len=len(self.tokenizer(prompt_text)["input_ids"])
        labels=[self.pad_token_id]*prompt_len+full_input_ids[prompt_len:]
        
        input_ids=full_input_ids[:-1]
        labels=labels[1:]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": pixel_values,
        }


class VLMDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        pixel_values = torch.stack(
            [feature.pop("pixel_values") for feature in features]
        )

        batch = super().__call__(features, return_tensors)

        batch["pixel_values"] = pixel_values
        return batch


if __name__ == "__main__":
    dataset = LoRADataset(qwen_path, clip_path)
    # print(random.choice(dataset))
