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
pair_input = [(this_dict['text'], this_dict['label']) for this_dict in data_rows]

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def collate_batch(batch):
    label_list, text_list = [], []
    texts = [_text for (_text, _label) in batch]
    labels = [_label for (_text, _label) in batch]
    label_list = torch.tensor(labels, dtype=torch.int64)
    text_list = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    return text_list, label_list.to(device)


dataloader = DataLoader(pair_input, batch_size=16, shuffle=True, collate_fn=collate_batch)


class BiasDetection(nn.Module):

    def __init__(self):
        super(BiasDetection, self).__init__()
        self.bert_embeddor = AutoModel.from_pretrained("bert-base-uncased")
        self.linear_l1 = nn.Linear(768, 256)
        self.linear_l2 = nn.Linear(256, 3)  # 3 classes

    def forward(self, input_article):
        # Embed all sentences - TODO
        embedded = self.bert_embeddor(**input_article)
        embedding_output = embedded[1]

        # Extract 1st token embedding
        # Either define linear layer here or pad all embeddings to fixed value - TODO
        linear_l1_output = self.linear_l1(embedding_output.view(-1, 768))
        linear_l2_output = self.linear_l2(linear_l1_output)
        return linear_l2_output


this_model = BiasDetection()
this_model.to(device)

lossfunction = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(this_model.parameters(), lr=0.01)

for epoch in range(0, 2):
    this_model.train()
    for article_batch, label_batch in dataloader:
        optimizer.zero_grad()
        model_output = this_model(article_batch)
        predicted_label = model_output.argmax(1)
        loss = lossfunction(model_output, label_batch)
        loss.backward()
        optimizer.step()
        accuracy = (predicted_label==label_batch).sum().item()/16
        print(f"Current Accuracy: {accuracy}")

print('Done!')
