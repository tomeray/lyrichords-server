# Define a linear classifier on top of the pre-trained model
import torch
from torch import nn


class ChordClassifier(nn.Module):
    def __init__(self, pretrained_model, num_classes):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.classifier = nn.Linear(pretrained_model.config.hidden_size, num_classes)

    def forward(self, input_values, attention_mask):
        outputs = self.pretrained_model(input_values, attention_mask, output_hidden_states=True)
        all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()

        hidden_states = outputs.last_hidden_state
        # Reshape hidden states to have shape [batch_size * time_steps, feature_dim]
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        # Apply linear classifier to obtain logits for each timestep
        logits = self.classifier(hidden_states)
        predictions = torch.softmax(logits, dim=-1)

        current_batch_size = input_values.shape[0]
        # Reshape logits to have shape [batch_size, time_steps, num_classes]
        logits = logits.view(current_batch_size, -1, logits.shape[-1])
        predictions = predictions.view(current_batch_size, -1, predictions.shape[-1])

        return {'logits': logits, 'predictions': predictions}
