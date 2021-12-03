import torch
from transformers import AutoTokenizer
from preprocess import PreprocessModel
from models.linearModel import BiasDetectionLinear
from models.cnnModel import BiasDetectionCnn


class Train:
    def __init__(self, batch_size=16, learning_rate=2e-5, epoch_size=20, print_every=25, sample_size=1000,
                 model_type='cnn', torch_device='cuda', n_filters=64):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.preprocess_obj = PreprocessModel(tokenizer, batch_size=batch_size, repop_db=False, sample_size=sample_size,
                                              torch_device=torch_device)
        self.train_dataloader, self.val_dataloader, self.test_dataloader = self.preprocess_obj.get_dataloader()
        self.model = self.return_model(model_type)(n_classes=2, n_filters=n_filters)
        self.model.to(self.preprocess_obj.device)
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        self.epoch_size = epoch_size
        self.print_every = print_every

    def return_model(self, model_type):
        dict_models_type = {
            'linear': BiasDetectionLinear,
            'cnn': BiasDetectionCnn
        }
        return dict_models_type[model_type]

    def train_model(self):
        for epoch in range(self.epoch_size):
            self.model.train()

            total_acc_train = 0
            total_loss_train = 0

            for batch_count, (article_batch, label_batch) in enumerate(self.train_dataloader):
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

                if batch_count % self.print_every == 0 and batch_count != 0:
                    print(f"Current Train Accuracy: {total_acc_train / batch_count: .3f}")

            total_acc_val, total_loss_val = self.val_model(self.val_dataloader)

            print(
                f'Epochs: {epoch + 1} | Train Loss: {total_loss_train / len(self.train_dataloader): .3f} \
                | Train Accuracy: {total_acc_train / len(self.train_dataloader): .3f} \
                | Val Loss: {total_loss_val / len(self.val_dataloader): .3f} \
                | Val Accuracy: {total_acc_val / len(self.val_dataloader): .3f}')

        torch.save(self.model.state_dict(), "bias-aware-model.pth")
        total_acc_test, total_loss_test = self.val_model(self.test_dataloader)

        print(
            f' Test Loss: {total_loss_test / len(self.test_dataloader): .3f} \
            | Test Accuracy: {total_acc_test / len(self.test_dataloader): .3f}')

        print('Done!')

    def val_model(self, dataloader_input):
        self.model.eval()

        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():
            for article_batch, label_batch in dataloader_input:
                model_output = self.model(input_ids=article_batch['input_ids'],
                                          attention_mask=article_batch['attention_mask'])
                predicted_label = model_output.argmax(1)

                accuracy = (predicted_label == label_batch).sum().item() / self.preprocess_obj.batch_size
                total_acc_val += accuracy

                loss = self.loss_function(model_output, label_batch)
                total_loss_val += loss.item()

        return total_acc_val, total_loss_val


# bias_aware = Train(batch_size=16, learning_rate=2e-5, epoch_size=2, print_every=25, sample_size=100, model_type='cnn',
#                    torch_device='cpu', n_filters=64)
# bias_aware.train_model()
