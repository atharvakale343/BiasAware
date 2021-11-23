import torch
from transformers import AutoTokenizer
from preprocess import PreprocessModel
from models.linearModel import BiasDetectionLinear


class Train:
    def __init__(self, batch_size=2, learning_rate=0.01):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.preprocess_obj = PreprocessModel(tokenizer, batch_size=batch_size, repop_db=False, sample_size=1000)
        self.train_dataloader, self.test_dataloader = self.preprocess_obj.get_dataloader()

        self.model = BiasDetectionLinear(n_classes=2)
        self.model.to(self.preprocess_obj.device)
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def train_model(self, nEpochs=2):
        for epoch in range(0, nEpochs):
            self.model.train()

            total_acc_train = 0
            total_loss_train = 0

            for article_batch, label_batch in self.train_dataloader:
                model_output = self.model(input_ids=article_batch['input_ids'],
                                          attention_mask=article_batch['attention_mask'])
                predicted_label = model_output.argmax(1)

                loss = self.loss_function(model_output, label_batch)
                total_loss_train += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                accuracy = (predicted_label == label_batch).sum().item() / self.preprocess_obj.batch_size
                total_acc_train += accuracy
                print(f"Current Batch Accuracy: {accuracy}")

            total_acc_val, total_loss_val = self.test_model()

            print(
                f'Epochs: {epoch + 1} | Train Loss: {total_loss_train / len(self.train_dataloader): .3f} \
                | Train Accuracy: {total_acc_train / len(self.train_dataloader): .3f} \
                | Val Loss: {total_loss_val / len(self.test_dataloader): .3f} \
                | Val Accuracy: {total_acc_val / len(self.test_dataloader): .3f}')

        torch.save(self.model.state_dict(), "bias-aware-model.pth")
        print('Done!')

    def test_model(self):
        self.model.eval()

        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():
            for article_batch, label_batch in self.test_dataloader:
                model_output = self.model(input_ids=article_batch['input_ids'],
                                          attention_mask=article_batch['attention_mask'])
                predicted_label = model_output.argmax(1)

                accuracy = (predicted_label == label_batch).sum().item() / self.preprocess_obj.batch_size
                total_acc_val += accuracy

                loss = self.loss_function(model_output, label_batch)
                total_loss_val += loss.item()

        return total_acc_val, total_loss_val


# bias_aware = Train(batch_size=2, learning_rate=0.01)
# bias_aware.train_model()
