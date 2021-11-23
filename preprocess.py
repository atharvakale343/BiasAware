from scripts.access_db import AccessDatabase
from scripts.create_db import CreateDatabase
from torch.utils.data import DataLoader
import torch


class PreprocessModel:
    def __init__(self, tokenizer, repop_db=False, sample_size=1000):
        if repop_db:
            self.create_database(sample_size)
        self.batch_size = 16
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tokenizer

    def create_database(self, sample_size):
        data_create = CreateDatabase()
        data_create.populate_db(sample_size)
        data_create.close_db_conn()

    def access_database(self):
        data_access = AccessDatabase()
        data_df = data_access.get_all_articles()
        return data_access.parse_df_to_dict(data_df)

    def preprocess_input(self, list_rows, exclude_labels=None):
        if not exclude_labels:
            exclude_labels = [1]

        return [(this_dict['text'], this_dict['label']) for this_dict in list_rows if
                this_dict['label'] not in exclude_labels]

    def collate_batch(self, batch):
        label_list, text_list = [], []
        texts = [_text for (_text, _label) in batch]
        labels = [_label for (_text, _label) in batch]
        label_list = torch.tensor(labels, dtype=torch.int64)
        text_list = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        return text_list, label_list.to(self.device)

    def get_dataloader(self):
        data_rows = self.access_database()
        pair_input = self.preprocess_input(data_rows)
        return DataLoader(pair_input, batch_size=self.batch_size,
                          shuffle=True, collate_fn=self.collate_batch)
