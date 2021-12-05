import os
import torch
from transformers.models.auto.tokenization_auto import AutoTokenizer
from models.linearModel import BiasDetectionLinear
from models.cnnModel import BiasDetectionCnn
from models.CNN_BiLSTM_Model import BiasDetectionCnnBiLSTM
from models.BiLSTM_Model import BiasDetectionBiLSTM
from preprocess import PreprocessModel
from sklearn import metrics
import seaborn as sb
import numpy as np


class RunModel:
    def __init__(self, model_type='linear', path_to_models='saved_models', device='gpu', batch_size=8,
                 sample_size=1000):
        self.ModelClasses = {
            'linear': BiasDetectionLinear,
            'cnn': BiasDetectionCnn,
            'cnn_biLSTM': BiasDetectionCnnBiLSTM,
            'biLSTM': BiasDetectionBiLSTM,
        }
        self.path_to_models = path_to_models
        self.paths = {
            model:
                os.path.join(path_to_models, f'bias-aware-model-{model}.pth')
            for model in self.ModelClasses
        }

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.preprocess_obj = PreprocessModel(tokenizer, batch_size=batch_size, repop_db=False,
                                              sample_size=sample_size, torch_device=device)

        self.device = self.preprocess_obj.device
        self.model_type = model_type

    def get_or_load_dataloader(self, load_dataloader, load_dataloader_file):
        if load_dataloader:
            return torch.load(os.path.join(self.path_to_models, load_dataloader_file))
        else:
            _, _, test_dataloader = self.preprocess_obj.get_dataloader()
            return test_dataloader

    def initialize_model(self):
        kwargs = {'n_classes': 2, 'n_filters': 64}
        this_model = self.ModelClasses[self.model_type](**kwargs)
        state_dict = torch.load(self.paths[self.model_type], map_location=self.device)
        this_model.load_state_dict(state_dict)
        return this_model

    def get_predictions(self, input_model, load_dataloader=False, load_dataloader_file='test.pth'):
        predicted_labels = []
        actual_labels = []
        input_model.eval()

        test_dataloader = self.get_or_load_dataloader(load_dataloader, load_dataloader_file)

        for batch_count, (article_batch, label_batch) in enumerate(test_dataloader):
            try:
                model_output = input_model(input_ids=article_batch['input_ids'],
                                           attention_mask=article_batch['attention_mask'])
            except RuntimeError as e:
                print(f'Error Encountered: {e}')
                continue
            predicted_labels.extend(model_output.argmax(1).tolist())
            actual_labels.extend(label_batch.tolist())

        return predicted_labels, actual_labels

    @staticmethod
    def get_metrics(predicted_labels, actual_labels):
        accuracy = sum(
            [pred_label == actual_label for pred_label, actual_label in zip(predicted_labels, actual_labels)]) / len(
            predicted_labels)

        print(f'Accuracy: {accuracy}')

        conf = metrics.confusion_matrix(predicted_labels, actual_labels)
        print("Confusion Matrix : \n", conf)

        sb.heatmap(conf / np.sum(conf), annot=True, fmt='0.2%', cmap='Reds')
