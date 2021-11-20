from scripts.access_db import AccessDatabase
from torch.utils.data import DataLoader
import torch
import nltk
from transformers import AutoTokenizer, AutoModel, TFAutoModel
from torch import nn

nltk.download('punkt')
batch_size = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_access = AccessDatabase()
data_df = data_access.get_all_articles()
data_rows = data_access.parse_df_to_dict(data_df)

# Preprocess training input
pair_input = [(nltk.sent_tokenize(this_dict['text']), this_dict['label']) for this_dict in data_rows]

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def collate_batch(batch):
    label_list, text_list = [], []
    for (_text, _label) in batch:
        label_list.append(_label)
        processed_text = [tokenizer(sentence, return_tensors="pt") for sentence in _text]
        text_list.append(processed_text)
    label_list = torch.tensor(label_list, dtype=torch.int64)
    return text_list, label_list.to(device)


dataloader = DataLoader(pair_input, batch_size=16, shuffle=True, collate_fn=collate_batch)


class BiasDetection(nn.Module):

    def __init__(self):
        super(BiasDetection, self).__init__()
        self.bert_embeddor = AutoModel.from_pretrained("bert-base-uncased")
        self.linear_layer = nn.Linear(10, 16)

    def forward(self, input_article):
        embedded = self.bert_embeddor(**input_article[0][0])
        embedding_output = embedded[0]
        print()


this_model = BiasDetection()

for article, label in dataloader:
    this_model(article)
