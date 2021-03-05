from bert4torch.layer import *
from data_preprocessCopy1 import *
from xtools import *
from loss import *
from bert4torch.lightmodel import *
from bert4torch.callback import Checkpoint
from bert4torch.evaluate import *

pl.seed_everything(128)

lr = 2e-5
max_len = 600
hidden_size = 160
num_heads = 8

model_path = 'roberta'
train = pd.read_csv('../train.csv')
train['1'] = train["1"].apply(eval)
test = pd.read_csv('../dev.csv')
test['1'] = test['1'].apply(eval)
loss_fct = nn.CrossEntropyLoss()


def load_vocab_from_pretrain(file):
    vocabs = []
    for idx, line in enumerate(open(file, encoding='utf-8')):
        if idx > 0 or line.count(' ') > 10:
            vocab, *vector = line.rstrip().split()
            vocabs.append(vocab)
    return vocabs


embedding_file = "/home/vocust001/xly/key_phrase_extraction/整个观点抽取模型/flat_lattice/sgns.merge.shrink.txt"
vocabs = load_vocab_from_pretrain(embedding_file)

processor = FLATDataProcessor(train['1'], vocabs)

train_data, _ = processor.batch_encode(train['0'], train['1'])
dataset = KeyDataset(train_data)
train_data = DataLoader(dataset, shuffle=True, batch_size=10, collate_fn=default_collate)

dev_data, offset = processor.batch_encode(test['0'], test['1'])
dataset = KeyDataset(dev_data)
dev_data = DataLoader(dataset, shuffle=False, batch_size=20, collate_fn=default_collate)


class FLATModel(nn.Module):
    def __init__(self, model_path, num_class, w2v_file):
        super().__init__()
        self.embedding = FLATEmbedding(model_path, w2v_file)
        self.flat = FLAT(max_len=max_len, hidden_size=hidden_size, num_heads=num_heads)
        self.crf = LinearCRF(num_class)

    def forward(self, batch):
        a = batch['input_ids']
        b = batch['word_ids']
        c = batch['attention_mask']
        d = batch['word_mask']
        batch['char_word_vec'] = self.embedding(a, b, c, d)
        logits = self.flat(batch)
        logits = self.crf(logits)
        return logits

    def compute_loss(self, logits, batch):
        return self.crf.compute_loss(logits, batch['labels'], batch['attention_mask'])


checkpoint = Checkpoint(ner_span_f1, predict_classical_ner, dev_data, test['1'], offset=offset,
                        id2label=processor.id2label,
                        texts=test['0'])
model = FLATModel(model_path, processor.num_class, embedding_file)

optimizer = torch.optim.Adam(model.parameters(), lr)

model2 = LightModel(model, optimizer)

trainer = pl.Trainer(max_epochs=10, gpus='1', logger=False, callbacks=[checkpoint])
trainer.fit(model2, train_data)
