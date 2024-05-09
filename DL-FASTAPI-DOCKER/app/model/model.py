from fastapi import FastAPI, Body, Request
from pathlib import Path
import torch
from torch import nn
from transformers import GPT2Model, GPT2Tokenizer

__version__ = "1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent


class SimpleGPT2SequenceClassifier(nn.Module):
    def __init__(self, hidden_size: int, num_classes:int ,max_seq_len:int, gpt_model_name:str):
        super(SimpleGPT2SequenceClassifier,self).__init__()
        self.gpt2model = GPT2Model.from_pretrained(gpt_model_name)
        self.fc1 = nn.Linear(hidden_size*max_seq_len, num_classes)

        
    def forward(self, input_id, mask):
        """
        Args:
                input_id: encoded inputs ids of sent.
        """
        gpt_out, _ = self.gpt2model(input_ids=input_id, attention_mask=mask, return_dict=False)
        batch_size = gpt_out.shape[0]
        linear_output = self.fc1(gpt_out.view(batch_size,-1))
        return linear_output

model_new = SimpleGPT2SequenceClassifier(hidden_size=768, num_classes=5, max_seq_len=128, gpt_model_name="gpt2")
model_new.load_state_dict(torch.load(f"{BASE_DIR}/gpt2-text-classifier-model-{__version__}.pt", map_location=torch.device('cpu')))
model_new.eval()

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token



def predict_pipeline(text):

    fixed_text = " ".join(text.lower().split())
      #print(fixed_text)
        
    model_input = tokenizer(fixed_text, padding='max_length', max_length=128, truncation=True, return_tensors="pt")

    mask = model_input['attention_mask'].cpu()
    input_id = model_input["input_ids"].squeeze(1).cpu()

    output = model_new(input_id, mask)
    prob = torch.nn.functional.softmax(output, dim=1)[0]

    labels_map = {
        0: "business",
        1: "entertainment",
        2: "sport",
        3: "tech",
        4: "politics"
             }

    pred_label = labels_map[output.argmax(dim=1).item()]
    
    return pred_label