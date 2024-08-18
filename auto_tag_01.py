from transformers import pipeline

# 初始化预训练模型
classifier = pipeline('text-classification', model='bert-base-uncased')

# 自动打标
text = "This is a critical system alert."
result = classifier(text)
print(f"Predicted Label: {result[0]['label']}")
