from transformers import AutoModelForCausalLM
import os
qwen=AutoModelForCausalLM.from_pretrained(os.path.join(os.path.dirname(os.path.abspath(__file__)),'Qwen3-0.6B'))
print(qwen)