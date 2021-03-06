from torch import nn
from transformers import AutoModel
import torch.nn.functional as F
import torch


class BiasDetectionBiLSTM(nn.Module):
    def __init__(self, dropout_c=0.5, n_classes=2, hidden_size=256, num_layers=2, bidirectional=True, n_filters=None):
        super(BiasDetectionBiLSTM, self).__init__()
        self.bert_embedding = AutoModel.from_pretrained("bert-base-uncased")
        self.lstm = nn.LSTM(input_size=768, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        self.dropout_l = nn.Dropout(dropout_c)
        self.linear_lf = nn.Linear(hidden_size*(bidirectional+1), n_classes)  # f{n_classes} classes

    def forward(self, input_ids, attention_mask):
        # sequence_output: [batch_size, seq_len, embedding_dim]
        # pooled_output : [batch_size, embedding_dim]
        sequence_output, pooled_output = self.bert_embedding(input_ids=input_ids, attention_mask=attention_mask,
                                                             return_dict=False)
        # lstm_output: [batch_size, seq_len, 2*hidden_size]
        lstm_output, (hidden, cell) = self.lstm(sequence_output)

        # last_LSTM_cell: [batch_size, 2*hidden_size]
        last_LSTM_cell = lstm_output[:,-1]

        dropout_l_output = self.dropout_l(last_LSTM_cell)

        # linear_l_output: [batch_size, n_classes]
        linear_l_output = self.linear_lf(dropout_l_output)

        return linear_l_output
