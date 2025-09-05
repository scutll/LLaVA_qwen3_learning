import os
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor
from safetensors.torch import load_file as load_safetensors


from model import VLM, VLMConfig

def generate_response():
    base_path = os.path.dirname(os.path.abspath(__file__))
    state_path = os.path.join(base_path, "output", "pretrained_best_model") 
    # state_path = os.path.join(base_path, "output", "checkpoint-400") 
    qwen_path = os.path.join(base_path, "Qwen3-0.6B")
    clip_path = os.path.join(base_path, "clip-vit-base-patch16") 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    print("加载模型配置...")
    config = VLMConfig(qwen_path=qwen_path, clip_path=clip_path, dtype=torch.bfloat16)
    
    print("加载模型结构...")
    model = VLM(config).to(device)
    model.eval()

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

    while True:
        try:
            image_path_input = input("请输入图片路径 (例如: data/test.jpg)，或按回车跳过: ")
            pixel_values = None
            if image_path_input.strip():
                image_path = os.path.join(base_path, image_path_input)
                if not os.path.exists(image_path):
                    print("图片路径不存在，请重试。")
                    continue
                image = Image.open(image_path).convert("RGB")
                pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(
                    device, dtype=model.config.dtype
                )

            prompt_input = input("请输入问题 (输入 'exit' 退出): ")
            if prompt_input.lower() == 'exit':
                print("程序退出。")
                break
            if not prompt_input.strip():
                print("问题不能为空。")
                continue
            
            if "<image>" not in prompt_input:
                prompt_input += "\n<image>"
            
            template_prompt = prompt_input.replace("<image>", "<|image_pad|>" * model.image_pad_num)
            
            # 使用与训练时一致的对话模板
            conversation = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": template_prompt}
            ]
            formatted_prompt = tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True,enable_thinking=False
            )
            input_ids = tokenizer(formatted_prompt, return_tensors="pt")["input_ids"].to(device)

            print("\n模型生成中...")
            initial_prompt_len = input_ids.shape[1]
            max_new_tokens = 256  
            eos_token_id = tokenizer.eos_token_id

            with torch.no_grad():
                for _ in range(max_new_tokens):
                    outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=None)
                    next_token_logits = outputs.logits[:, -1, :]
                    next_token_id = torch.argmax(next_token_logits, dim=-1)
                    
                    # 检查是否生成了结束符
                    if next_token_id.item() == eos_token_id:
                        print("(检测到结束符)")
                        break
                    
                    input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=1)

            response_ids = input_ids[:, initial_prompt_len:]
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