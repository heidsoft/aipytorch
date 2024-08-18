import torch
import torch.nn as nn
import torch.optim as optim

# 定义LSTM模型
class CharLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_dim, n_layers):
        super(CharLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # LSTM层
        self.lstm = nn.LSTM(vocab_size, hidden_dim, n_layers, batch_first=True)
        
        # 全连接层
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        out = out.reshape(out.size(0) * out.size(1), self.hidden_dim)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        return hidden

# 准备数据
text = "hello world"
chars = sorted(list(set(text)))
vocab_size = len(chars)

char_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_char = {i: ch for i, ch in enumerate(chars)}

# 将文本转换为整数序列
input_seq = [char_to_int[ch] for ch in text[:-1]]
target_seq = [char_to_int[ch] for ch in text[1:]]

# 将数据转换为 one-hot 编码
def one_hot_encode(sequence, n_labels):
    one_hot = torch.zeros(len(sequence), n_labels)
    for i, value in enumerate(sequence):
        one_hot[i][value] = 1.0
    return one_hot

input_seq = one_hot_encode(input_seq, vocab_size).unsqueeze(0)
target_seq = torch.tensor(target_seq)

# 定义超参数
hidden_dim = 12
n_layers = 1
lr = 0.01
n_epochs = 200

# 初始化模型、损失函数和优化器
model = CharLSTM(vocab_size, hidden_dim, n_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# 训练模型
for epoch in range(1, n_epochs + 1):
    hidden = model.init_hidden(1)
    
    optimizer.zero_grad()
    output, hidden = model(input_seq, hidden)
    loss = criterion(output, target_seq)
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f'Epoch: {epoch} | Loss: {loss.item():.4f}')

# 生成文本
def predict(model, char, hidden=None):
    x = one_hot_encode([char_to_int[char]], vocab_size).unsqueeze(0)
    out, hidden = model(x, hidden)
    prob = nn.functional.softmax(out, dim=1).data
    char_ind = torch.max(prob, dim=1)[1].item()
    return int_to_char[char_ind], hidden

# 从模型中生成新字符序列
def sample(model, out_len, start='h'):
    model.eval()
    chars = [start]
    hidden = model.init_hidden(1)
    for _ in range(out_len):
        char, hidden = predict(model, chars[-1], hidden)
        chars.append(char)
    return ''.join(chars)

print("生成的文本: ")
print(sample(model, 10, start='h'))
