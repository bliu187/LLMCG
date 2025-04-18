import numpy as np
import torch
from transformers import BertTokenizer
import pandas as pd
from torch import nn
from transformers import BertModel
from torch.optim import Adam
from tqdm import tqdm
import json
import os

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.labels = [labels[label] for label in df['category']]
        self.texts = [tokenizer(text,
                                padding='max_length',
                                max_length=512,
                                truncation=True,
                                return_tensors="pt")
                      for text in df['text']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y

class BertClassifier(nn.Module):
    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 5)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer, pooled_output


def train(model, train_data, val_data, learning_rate, epochs):
    # 通过Dataset类获取训练和验证集
    train, val = Dataset(train_data), Dataset(val_data)
    # DataLoader根据batch_size获取数据，训练时选择打乱样本
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)
    # 判断是否使用GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    # 开始进入训练循环
    for epoch_num in range(epochs):
        # 定义两个变量，用于存储训练集的准确率和损失
        total_acc_train = 0
        total_loss_train = 0
        # 进度条函数tqdm
        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)
            # 通过模型得到输出
            output, cls_emb = model(input_id, mask)
            # 计算损失
            train_label = train_label.long()   # added
            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()
            # 计算精度
            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc
            # 模型更新
            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
        total_acc_val = 0
        total_loss_val = 0
        with torch.no_grad():
            # 循环获取数据集，并用训练好的模型进行验证
            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)
                output, cls_emb = model(input_id, mask)
                val_label = train_label.long()  # added
                batch_loss = criterion(output, val_label)
                total_loss_val += batch_loss.item()
                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        print(
            f'''Epochs: {epoch_num + 1} 
              | Train Loss: {total_loss_train / len(train_data): .3f} 
              | Train Accuracy: {total_acc_train / len(train_data): .3f} 
              | Val Loss: {total_loss_val / len(val_data): .3f} 
              | Val Accuracy: {total_acc_val / len(val_data): .3f}''')

def evaluate(model, test_data):
    test = Dataset(test_data)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        model = model.cuda()
    total_acc_test = 0
    with torch.no_grad():
        for test_input, test_label in test_dataloader:
            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)
            output, cls_emb = model(input_id, mask)
            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc
    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')


if __name__ == '__main__':
    model_name = './bert-base-uncased'
    file_path = "./Finetuned_LM/LLMCG_Rel_LM.pth"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    labels = {'NA': 0, '/people/person/place_of_birth': 1, '/people/deceased_person/place_of_death': 2, '/people/person/education./education/education/degree': 3, '/people/person/education./education/education/institution': 4}
    # 拆分训练集、验证集和测试集 8:1:1
    with open("./GDS/GDS_Relation.json", "r", encoding="utf-8") as file:
        data_ori = json.load(file)
    df = pd.DataFrame(data_ori)
    np.random.seed(112)
    df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42), [int(.8 * len(df)), int(.9 * len(df))])
    print(len(df_train), len(df_val), len(df_test))

    if os.path.exists(file_path) and os.path.isfile(file_path):
        print("Finetuned model is existed")
        model = BertClassifier()
        model.load_state_dict(torch.load(file_path))
        evaluate(model, df_test)

        # obtain the vectot of relation or entity type embeddings
        with open("./GDS/GDS_relation_text.json", "r", encoding="utf-8") as file:
            data_tar = json.load(file)
        df_tar = pd.DataFrame(data_tar)
        tar_vec = Dataset(df_tar)
        tar_dataloader = torch.utils.data.DataLoader(tar_vec, batch_size=1)
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        if use_cuda:
            model = model.cuda()
        with torch.no_grad():
            tar_rep = torch.empty(0, 768).to(device)
            tar_label = torch.empty(0).to(device)
            for tar_input, tar_cate in tar_dataloader:
                tar_cate = tar_cate.to(device)
                mask = tar_input['attention_mask'].to(device)
                input_id = tar_input['input_ids'].squeeze(1).to(device)
                output, cls_emb = model(input_id, mask)
                tar_rep = torch.cat((tar_rep, cls_emb), dim=0)
                tar_label = torch.cat((tar_label, tar_cate), dim=0)
            label_list = tar_label.tolist()
            tar_rep = tar_rep.tolist()
            label_list = [int(x) for x in label_list]
            reversed_labels = {value: key for key, value in labels.items()}
            label_list = [reversed_labels[i] for i in label_list]
            tar_data = [{"vector": vec, "category": cat} for vec, cat in zip(tar_rep, label_list)]
            with open("relation_vectors_with_categories.json", "w", encoding="utf-8") as f:
                json.dump(tar_data, f, ensure_ascii=False)

    else:
        print("Finetuned is not existed")
        EPOCHS = 1
        model = BertClassifier()
        LR = 1e-6
        train(model, df_train, df_val, LR, EPOCHS)
        torch.save(model.state_dict(), file_path)
        model = BertClassifier()
        model.load_state_dict(torch.load(file_path))
        evaluate(model, df_test)
