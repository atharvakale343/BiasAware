from torch import nn
from transformers import AutoModel


class BiasDetectionLinear(nn.Module):

    def __init__(self, dropout_c=0.5, n_classes=2):
        super(BiasDetectionLinear, self).__init__()
        self.bert_embedding = AutoModel.from_pretrained("bert-base-uncased")
        self.dropout_l = nn.Dropout(dropout_c)
        self.linear_lf = nn.Linear(768, n_classes)  # f{n_classes} classes

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert_embedding(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        dropout_l_output = self.dropout_l(pooled_output)
        linear_l_output = self.linear_lf(dropout_l_output)
        return linear_l_output
