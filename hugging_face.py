from transformers import BertTokenizer, BertForSequenceClassification
import torch

# 初始化 BERT 模型和 tokenizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 定义输入文本并进行推理
inputs = tokenizer("这是一个示例文本", return_tensors="pt", padding=True, truncation=True, max_length=512)
outputs = model(**inputs)
logits = outputs.logits
prediction = torch.argmax(logits, dim=-1)

print(f"预测标签: {prediction.item()}")
