import os
import torch
from transformers.models.auto.tokenization_auto import AutoTokenizer
from models.linearModel import BiasDetectionLinear
from models.cnnModel import BiasDetectionCnn
from models.CNN_BiLSTM_Model import BiasDetectionCnnBiLSTM
from models.BiLSTM_Model import BiasDetectionBiLSTM
from preprocess import PreprocessModel
from torch.utils.data import ConcatDataset, DataLoader

ModelClasses = {
    'linear': BiasDetectionLinear,
    'cnn': BiasDetectionCnn,
    'cnn_biLSTM': BiasDetectionCnnBiLSTM,
    'biLSTM': BiasDetectionBiLSTM,
}

paths = {
        model: 
            os.path.join('saved_models', f'bias-aware-model-{model}.pth') 
        for model in ModelClasses
}

device = 'cpu'

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
preprocess_obj = PreprocessModel(tokenizer, batch_size=16, repop_db=False, sample_size=1000,
                                              torch_device=device)

device = preprocess_obj.device

_, _, test_dataloader = preprocess_obj.get_dataloader()



model_type = 'linear'

kwargs = {'n_classes': 2, 'n_filters': 64}

model = ModelClasses[model_type](**kwargs)

state_dict = torch.load(paths[model_type], map_location=device)

model.load_state_dict(state_dict)

predicted_labels = []
actual_labels = []
model.eval()

for batch_count, (article_batch, label_batch) in enumerate(test_dataloader):
    
    model_output = model(input_ids=article_batch['input_ids'],
                                    attention_mask=article_batch['attention_mask'])

    predicted_labels.extend(model_output.argmax(1))
    actual_labels.extend(label_batch)
    print()

print()

