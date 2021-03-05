from bert4torch.layer import *
from data_preprocessCopy1 import *
from xtools import *
from loss import *
from bert4torch.lightmodel import *
from bert4torch.callback import Checkpoint
from bert4torch.evaluate import *

pl.seed_everything(128)

lr = 2e-5
model_path = 'roberta'
train = pd.read_csv('../train.csv')
train['1'] = train["1"].apply(eval)
test = pd.read_csv('../dev.csv')
test['1'] = test['1'].apply(eval)
loss_fct = nn.CrossEntropyLoss()

processor = BiaffineDataProcessor(train['1'])

train_data, _ = processor.batch_encode(train['0'], train['1'])
dataset = KeyDataset(train_data)
train_data = DataLoader(dataset, shuffle=True, batch_size=10, collate_fn=biaffine_collate)

dev_data, offset = processor.batch_encode(test['0'], test['1'])
dataset = KeyDataset(dev_data)
dev_data = DataLoader(dataset, shuffle=False, batch_size=20, collate_fn=default_collate)


class BiAffineNer(nn.Module):
    def __init__(self, model_path, num_class):
        super().__init__()
        self.bert = get_bert_model(model_path)
        self.biaffine = BiAffine(num_class, LabelSmoothingCrossEntropy())

    def forward(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        hidden = self.bert(input_ids, attention_mask)[0]
        logits = self.biaffine(hidden)
        return logits

    def compute_loss(self, logits, batch):
        return self.biaffine.compute_loss(logits, batch)


checkpoint = Checkpoint(ner_span_f1, predict_biaffine, dev_data, test['1'], offset=offset, id2label=processor.id2label,
                        texts=test['0'])
model = BiAffineNer(model_path, processor.num_class)

optimizer = torch.optim.Adam(model.parameters(), lr)

model2 = LightModel(model, optimizer)

trainer = pl.Trainer(max_epochs=10, gpus=1, logger=False, callbacks=[checkpoint])
trainer.fit(model2, train_data)
