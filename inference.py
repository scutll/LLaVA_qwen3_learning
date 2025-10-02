import os
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor
from safetensors.torch import load_file as load_safetensors
from path_config import qwen_config_path, clip16_config_path

from model import VLM, VLMConfig

# 跟ai诗句生成一样，其实就是要去制作一个generater来控制模型的返回句子或者说是形式
# 这还只是单论对话的实现,多轮对话的功能还待进一步
# 其实最核心的就只是将input_ids丢进模型然后获取一个新token,一直循环直到达到极限长度或者是<eos>
def generate_response():

    base_path = os.path.dirname(os.path.abspath(__file__))
    # 模型目录
    state_path = os.path.join(base_path, "stage1_patch16_loss3p9", "pretrained_best_model") 
    # state_path = os.path.join(base_path, "output", "checkpoint-400") 
    # qwen和clip模型路径，用于加载预训练模型的tokenizer
    qwen_path = os.path.join(base_path, qwen_config_path)
    clip_path = os.path.join(base_path, clip16_config_path) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    print("加载模型配置...")
    config = VLMConfig(qwen_path=qwen_path, clip_path=clip_path, dtype=torch.bfloat16)
    
    print("加载模型结构...")
    model = VLM(config).to(device)
    model.eval()

    # 加载VLM模型
    checkpoint_file = os.path.join(state_path, "model.safetensors")
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"在 {state_path} 中找不到权重文件 model.safetensors !")
    
    print(f"从 {checkpoint_file} 加载权重...")
    state_dict = load_safetensors(checkpoint_file)
    model.load_state_dict(state_dict, strict=False)
    

    print("加载分词器和图像处理器...")
    tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(clip_path)
    print("模型加载完成，进入交互模式。")

    
    # 正式接收内容并推理的环节
    while True:
        try:
            # 输入图片
            image_path_input = input("请输入图片路径 (例如: data/test.jpg)，或按回车跳过: ")
            pixel_values = None
            # strip去掉开头和结尾的空格、换行符、制表符等
            if image_path_input.strip():
                image_path = os.path.join(base_path, image_path_input)
                if not os.path.exists(image_path):
                    print("图片路径不存在，请重试。")
                    continue
                # 处理图片
                image = Image.open(image_path).convert("RGB")
                pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(
                    device, dtype=model.config.dtype
                )
                # pixel_values的形状是(batch_size, 3, 224, 224)
            # 输入问题
            prompt_input = input("请输入问题 (输入 'exit' 退出): ")
            if prompt_input.lower() == 'exit':
                print("程序退出。")
                break
            if not prompt_input.strip():
                print("问题不能为空。")
                continue
            
            # 添加图片占位符
            if "<image>" not in prompt_input:
                prompt_input += "\n<image>"
            
            # <image>替换成49(patch32)个<|image_pad|>
            template_prompt = prompt_input.replace("<image>", "<|image_pad|>" * model.image_pad_num)
            
            # 使用与训练时一致的对话模板
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
            # formatted_prompt还是string
            # print(formatted_prompt)  
            
            # 转化成token
            input_ids = tokenizer(formatted_prompt, return_tensors="pt")["input_ids"].to(device)
            # input_ids -> (batch_size, L)
            print("\n模型生成中...")
            # 获取prompt长度
            initial_prompt_len = input_ids.shape[1]
            # 限制生成token数
            max_new_tokens = 256  
            
            # end of sentense <eos>的id
            eos_token_id = tokenizer.eos_token_id


            with torch.no_grad():
                # 一个一个token生成
                for _ in range(max_new_tokens):
                    
                    # 最核心的生成环节：
                    # 每次都将token_ids放进模型,进行最新一个token的预测,然后将新token放在token_ids的末尾,再放进模型预测下一个token,一直直到生成最大长度或者是遇到结束符
                    outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=None)
                    
                    # 输出是logits[-1](batch_size(1), vocab_size)
                    # 用argmax来选出最有可能的token
                    # 思考: 如果不选择最高可能的token, 而在前k个可能token中进行随机选择,模型的回答可能会更有多样性吗?
                    next_token_logits = outputs.logits[:, -1, :]
                    next_token_id = torch.argmax(next_token_logits, dim=-1)
                    
                    # 检查是否生成了结束符
                    if next_token_id.item() == eos_token_id:
                        print("(检测到结束符)")
                        break
                    
                    # 将新的token cat到input_ids后面
                    input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=1)

            # 将新生成的token(也就是回答)取出来作为输出
            response_ids = input_ids[:, initial_prompt_len:]
            # 通过tokenizer的decode将id映射回文字
            response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)

            print("-" * 30)
            print("模型回答:", response_text)
            print("-" * 30 + "\n")
            
        except KeyboardInterrupt:
            print("\n程序被手动中断。")
            break
        except Exception as e:
            print(f"发生错误: {e}")

if __name__ == "__main__":
    generate_response()