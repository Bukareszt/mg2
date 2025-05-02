import torch
import torch.nn as nn
from transformers import BertModel

FLAG_BERT_TUNING = False
FLAG_VICUNA_DATA_ONLY = False

class BasicBertForRegression(nn.Module):
    def __init__(self, model_name='bert-base-uncased', hidden_dim=128):
        super(BasicBertForRegression, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        

        self.cls = nn.Linear(config.hidden_size, hidden_dim)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]

        x = self.relu(self.cls(cls_output))

        x = self.relu(self.fc1(x))
        prediction = self.fc2(x).squeeze(-1)
        return prediction
