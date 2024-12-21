from transformers import pipeline
import csv

# 初始化预训练模型
classifier = pipeline('text-classification', model='distilbert-base-uncased-finetuned-sst-2-english')

# 要打标签的文本列表
texts = [
    "This is a critical system alert.",
    "The weather is sunny today.",
    "I love programming in Python.",
    "The stock market is volatile."
]

# 打标签并保存结果到CSV文件
with open('tagged_data.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Text', 'Predicted Label'])

    for text in texts:
        result = classifier(text)
        writer.writerow([text, result[0]['label']])

print("标签已保存到 tagged_data.csv 文件中")
