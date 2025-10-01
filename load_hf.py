import os
from transformers import AutoTokenizer, AutoProcessor, AutoModel
from datasets import load_dataset, DownloadConfig


os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

# -------------------- 配置路径 --------------------
base_path = os.path.dirname(os.path.abspath(__file__))
qwen_path = os.path.join(base_path, "Qwen3-0.6B")
clip_path = os.path.join(base_path, "clip-vit-base-patch16")
clip32_path = os.path.join(base_path, "clip-vit-base-patch32")
dataset_path = os.path.join(base_path, "data")
os.makedirs(qwen_path, exist_ok=True)
os.makedirs(clip_path, exist_ok=True)

# -------------------- Huggingface 模型配置 --------------------
qwen_name = "Qwen/Qwen3-0.6B"                  # Qwen 模型
clip_name = "openai/clip-vit-base-patch16"     # CLIP 模型
clip32_name = "openai/clip-vit-base-patch32"     # CLIP32 模型

# -------------------- 下载 Qwen tokenizer 和模型 --------------------
print("下载 Qwen tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(qwen_name, cache_dir=qwen_path, trust_remote_code=True)
print(tokenizer)
del tokenizer 

print("下载 Qwen 模型 ...")
qwen_model = AutoModel.from_pretrained(qwen_name, cache_dir=qwen_path, trust_remote_code=True)
print(qwen_model.config)
del qwen_model

# -------------------- 下载 CLIP tokenizer/processor 和模型 --------------------
print("下载 CLIP processor ...")
processor = AutoProcessor.from_pretrained(clip32_name, cache_dir=clip32_path)
print(processor)
del processor

print("下载 CLIP processor ... patch32")
processor = AutoProcessor.from_pretrained(clip_name, cache_dir=clip_path)
print(processor)
del processor

print("下载 CLIP 模型 ...")
clip_model = AutoModel.from_pretrained(clip_name, cache_dir=clip_path)
print(clip_model.config)
del clip_model
# -------------------- 下载数据集 --------------------
# 可以按需换成其他数据集
VLM_dataset_name = "tejasvaidhya/llava-cc3m-pretrain-595K"
download_config = DownloadConfig(resume_download=True)

print("下载VLM数据集:")
raw_dataset = load_dataset(
    VLM_dataset_name,
    split="train",
    cache_dir=dataset_path,
    download_config=download_config
)


LoRA_dataset = "trl-lib/llava-instruct-mix"
print('下载LoRA数据集: ')
Lora_dataset = load_dataset(
    LoRA_dataset,
    split="train",
    cache_dir=dataset_path,
    download_config=download_config
)
# 拆分验证集
split_dataset = raw_dataset.train_test_split(test_size=0.05, seed=42)
train_dataset = split_dataset["train"]
val_dataset = split_dataset["test"]
print(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")

print("下载完成，模型和数据集已保存到本地。")
