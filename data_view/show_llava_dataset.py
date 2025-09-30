# 用来训练VLM的数据集
# 这个数据集的形式就是image(bytes)+input(指令，请解释这一个图片里面的内容)+output(图片描述)

from datasets import load_dataset
from PIL import Image
import io
dataset = load_dataset('parquet',data_files='llava-cc3m-pretrain-595K_0.parquet')
sample = dataset['train'][7]

image = sample['image']
instruction = sample['input']
answer = sample['output']

image = Image.open(io.BytesIO(image))
image.show()

print("input:\n", instruction)
print("output: \n", answer)
