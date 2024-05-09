
import torch
from torch import nn
from transformers import GPT2Model, GPT2Tokenizer
from ModelClass import SimpleGPT2SequenceClassifier


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



def load_model(model_path):

 # define parameters andload model

  # load trained model
  model_new = SimpleGPT2SequenceClassifier(hidden_size=768, num_classes=5, max_seq_len=128, gpt_model_name="gpt2")
  model_new.load_state_dict(torch.load(model_path))
  model_new.eval()

  return model_new



def prediction(input_text, model_new):
  
  fixed_text = " ".join(input_text.lower().split())
  #print(fixed_text)

  tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
  tokenizer.padding_side = "left"
  tokenizer.pad_token = tokenizer.eos_token

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


example_text = """
The UK has accused President Putin of plotting to install a pro-Moscow figure to lead Ukraine's government.

The Foreign Office took the unusual step of naming former Ukrainian MP Yevhen Murayev as a potential Kremlin candidate.

Russia has moved 100,000 troops near to its border with Ukraine but denies it is planning an invasion.

UK ministers have warned that the Russian government will face serious consequences if there is an incursion.

In a statement, Foreign Secretary Liz Truss said: "The information being released today shines a light on the extent of Russian activity designed to subvert Ukraine, and is an insight into Kremlin thinking.

"Russia must de-escalate, end its campaigns of aggression and disinformation, and pursue a path of diplomacy."

The Russian Ministry of Foreign Affairs tweeted that the Foreign Office was "circulating disinformation" and urged it to "cease these provocative activities" and "stop spreading nonsense".

"""


model_path = "../model//gpt2-text-classifier-model.pt"
model_new = load_model(model_path)
pred_label = prediction(example_text, model_new)
# now send the pred label as a JSON response
