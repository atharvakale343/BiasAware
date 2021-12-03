from torch import nn
from transformers import AutoModel
import torch.nn.functional as F
import torch


class BiasDetectionCnnBiLSTM(nn.Module):
    def __init__(self, dropout_c=0.5, n_classes=2, n_filters=64, hidden_size=256, num_layers=2, bidirectional=True, list_kernel_sizes=(10, 50, 100)):
        super(BiasDetectionCnnBiLSTM, self).__init__()
        self.bert_embedding = AutoModel.from_pretrained("bert-base-uncased")
        self.lstm = nn.LSTM(input_size=768, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.convolutions = nn.ModuleList([nn.Conv1d(hidden_size*(bidirectional+1), n_filters, K) for K in list_kernel_sizes])
        self.relu_act = nn.ReLU()
        self.dropout_l = nn.Dropout(dropout_c)
        self.linear_lf = nn.Linear(len(list_kernel_sizes) * n_filters, n_classes)  # f{n_classes} classes

    def forward(self, input_ids, attention_mask):
        # sequence_output: [batch_size, seq_len, embedding_dim]
        # pooled_output : [batch_size, embedding_dim]
        sequence_output, pooled_output = self.bert_embedding(input_ids=input_ids, attention_mask=attention_mask,
                                                             return_dict=False)
        # lstm_output: [batch_size, seq_len, 2*hidden_size]
        lstm_output, (hidden, cell) = self.lstm(sequence_output)

        # Change lstm_output to [batch size, 2*hidden_size, seq len]
        permuted_output = lstm_output.permute(0, 2, 1)

        # conv_outputs: [batch_size, num_filters, seq_len - filter_sizes[i] + 1]
        conv_outputs = [self.relu_act(conv(permuted_output)) for conv in self.convolutions]

        # max_pooled_out: [batch_size, num_filters]
        max_pooled_out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in conv_outputs]

        # concatenated_out: [batch_size, num_filters * len(list_kernel_sizes)]
        concatenated_out = (torch.cat(max_pooled_out, dim=-1))

        dropout_l_output = self.dropout_l(concatenated_out)

        # linear_l_output: [batch_size, n_classes]
        linear_l_output = self.linear_lf(dropout_l_output)

        return linear_l_output
