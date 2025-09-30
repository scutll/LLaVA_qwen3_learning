from datasets import load_dataset
from PIL import Image
import io

dataset = load_dataset('parquet',data_files='train-00000-of-00032.parquet')

for k in dataset['train'][0].keys():
    print(k)

sample = dataset['train'][7]

image = sample['images']
print(len(image))
# print(image)
# image = Image.open(io.BytesIO(image))
image[0].show()

# 专门为了训练多轮对话能力的数据集

# 几轮对话和最后一个问题
print(sample['prompt'])
# 最后一个问题的回答
print(sample['completion'])