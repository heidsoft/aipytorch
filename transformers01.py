from transformers import BertTokenizer, BertForSequenceClassification
import torch

# 初始化 BERT 模型和 tokenizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 输入文本
inputs = tokenizer("I love machine learning!", return_tensors="pt", padding=True, truncation=True, max_length=512)
outputs = model(**inputs)

# 获取预测结果
logits = outputs.logits
prediction = torch.argmax(logits, dim=-1)
print(f"预测标签: {prediction.item()}")
