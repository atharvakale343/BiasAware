import torch
from transformers import AutoTokenizer
from preprocess import PreprocessModel
from models.linearModel import BiasDetectionLinear


class Train:
    def __init__(self):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        preprocess_obj = PreprocessModel(tokenizer, repop_db=False)
        self.dataloader = preprocess_obj.get_dataloader()

        self.model = BiasDetectionLinear()
        self.model.to(preprocess_obj.device)
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

    def train_model(self, nEpochs=2):
        for epoch in range(0, nEpochs):
            self.model.train()
            for article_batch, label_batch in self.dataloader:
                self.optimizer.zero_grad()
                model_output = self.model(article_batch)
                predicted_label = model_output.argmax(1)
                loss = self.loss_function(model_output, label_batch)
                loss.backward()
                self.optimizer.step()
                accuracy = (predicted_label == label_batch).sum().item() / 16
                print(f"Current Accuracy: {accuracy}")
        torch.save(self.model.state_dict(), "bias-aware-model.pth")
        print('Done!')
