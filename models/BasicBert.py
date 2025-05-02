import torch.nn as nn
from transformers import BertTokenizer, BertModel

class BasicBertForRegression(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super(BasicBertForRegression, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        prediction = self.regressor(cls_output)
        return prediction.squeeze(-1)
