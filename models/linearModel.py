from torch import nn
from transformers import AutoModel


class BiasDetectionLinear(nn.Module):

    def __init__(self, dropout_c=0.5, n_classes=2):
        super(BiasDetectionLinear, self).__init__()
        self.bert_embeddor = AutoModel.from_pretrained("bert-base-uncased")
        self.dropout_l = nn.Dropout(dropout_c)
        self.linear_l1 = nn.Linear(768, 256)
        self.linear_l2 = nn.Linear(256, n_classes)  # f{n_classes} classes
        self.relu_act = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert_embeddor(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        dropout_l_output = self.dropout_l(pooled_output)
        linear_l1_output = self.linear_l1(dropout_l_output)
        linear_l2_output = self.linear_l2(linear_l1_output)
        activated_output = self.relu_act(linear_l2_output)
        return activated_output
