import os
import torch
from transformers import Trainer, TrainingArguments, AutoTokenizer


from data.dataset import VLMDataset, VLMDataCollator
from model import VLM, VLMConfig
from path_config import qwen_config_path, clip16_config_path

base_path = os.path.dirname(os.path.abspath(__file__))
# qwen和clip的权重路径, 训练结果的输出路径
qwen_path = os.path.join(base_path, qwen_config_path)
clip_path = os.path.join(base_path, clip16_config_path)
output_dir = os.path.join(base_path, "output")

if __name__ == "__main__":
    TRAINING_TYPE=torch.bfloat16
    config = VLMConfig(qwen_path=qwen_path, clip_path=clip_path,torch_dtype=TRAINING_TYPE)
    model = VLM(config).cuda()
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(qwen_path)

    # model.qwen.gradient_checkpointing_enable(use_cache=False)
    # model.clip.gradient_checkpointing_enable(use_cache=False)

    print("模型和分词器加载完成，正在进行数据集构建...")
    train_dataset = VLMDataset(qwen_path, clip_path, config, split_type="train")
    eval_dataset = VLMDataset(qwen_path, clip_path, config, split_type="val")

    # 用于将多条数据拼成一个batch
    # 还负责长度padding对齐
    data_collator = VLMDataCollator(
        tokenizer=tokenizer, model=model, label_pad_token_id=tokenizer.pad_token_id, pad_to_multiple_of=8
    )

    print("配置参数...")
    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        # 训练/验证的batch大小
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        # 梯度累计步数,每累计训练4步后再更新参数,这样就相当于batch=32的训练
        gradient_accumulation_steps=4,
        # 学习率/轮次/权重衰减/优化器
        learning_rate=1e-4,
        num_train_epochs=1,
        weight_decay=0.05,
        optim="adamw_torch",  # adamw_8bit备用
        # 前5%的step线性增加学习率,也就是一开始从0提高到1e-4,然后之后就按照cosine进行衰减
        warmup_ratio=0.05,
        # 使用cosine decay作为学习率的调度衰减策略
        lr_scheduler_type="cosine",
        
        # bfloat16保留和float32一样的指数位,但缩短尾数,可以有更大的数值范围
        bf16=True,
        fp16=False,
        # 日志 每400steps记录一次训练指标
        logging_steps=400,
        logging_dir=os.path.join(output_dir, "logs"),
        
        # 保存策略 按step保存
        save_strategy="steps",
        eval_strategy="steps",
        # 最多保存两个checkpoints
        save_total_limit=2,
        # 每400steps保存和评估一次
        save_steps=400,
        eval_steps=400,
        
        # 训练结束加载loss最好的模型
        load_best_model_at_end=True,
        # 最好的模型以eval_loss作为指标
        metric_for_best_model="eval_loss",
        # 越小越好
        greater_is_better=False,
        # 训练过程中把日志输出到 TensorBoard
        report_to="tensorboard",
        # 数据加载核心数
        dataloader_num_workers=24,
        dataloader_pin_memory=True,
    )

    print("实例化Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    print("开始训练...")
    # 从最近的checkpoint开始训练
    # trainer.train(resume_from_checkpoint=True)
    trainer.train()

    print("训练完成，正在保存最佳模型...")
    trainer.save_model(os.path.join(output_dir, "pretrained_best_model"))
