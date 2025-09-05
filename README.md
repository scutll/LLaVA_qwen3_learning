# Qwen3-0.6B-VLM
利用Qwen3-0.6B和CLIP-ViT-B/16构建的多模态大模型，支持图文输入和单图多轮对话
整个过程只需要个位数的成本(前提限制在租用GPU)以及几小时时间，在可训练参数仅不到5M的情况下，即可表现出优秀的图片理解能力
本项目致力于提供一个低成本、易上手的多模态大模型训练和推理方案，供学习使用

## 项目中涉及的论文
LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS
Visual Instruction Tuning

## 环境配置
python 3.11.13(Conda)
cuda 12.4或更高
torch: 2.6.0+cu124
transformers: 4.56.1 
显存要求:16GB+用于训练(当前配置在RTX 5060ti-16G上完成，调整参数可降低显存需求)，至少4GB用于推理(如果使用GPU)

## 数据：

对齐：https://huggingface.co/datasets/tejasvaidhya/llava-cc3m-pretrain-595K

微调：https://huggingface.co/datasets/trl-lib/llava-instruct-mix
## 预训练模型

### 大语言模型

https://huggingface.co/Qwen/Qwen3-0.6B

### 视觉主干
https://huggingface.co/openai/clip-vit-base-patch16
https://huggingface.co/openai/clip-vit-base-patch32

## 训练
python vlm_train.py 用于预训练
python peft_train.py 用于微调

## 推理
python inference.py 用于预训练模型推理
python peft_inference.py 用于微调模型推理

## 注意事项
由于使用了peft库。微调后保存的lora_best_model将只含有lora层的权重参数，需要手动去输出目录寻找一个checkpoint，其中的model.safetensors才是完整的模型权重(包含llava投影层和lora微调层),否则得到的是仅含有lora层权重的模型，无llava层
admaw_8bit是显存不足时才使用的优化器，显存充足使用默认的AdamW即可
patch32和patch16都是可选的，但是在训练时img_pad_num数量需要调整
大语言模型可换用deepseek、llama等，只要特殊token和投影层对应上即可
