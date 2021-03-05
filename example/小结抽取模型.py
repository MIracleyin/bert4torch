from bert4torch.layer import *
from data_preprocess import *
from xtools import *
from loss import *
from bert4torch.lightmodel import *
from bert4torch.callback import Checkpoint
from bert4torch.evaluate import *
from bert4torch.model import *

file = '/home/vocust001/xly/ccc/extract_train.json'

data = []
for line in open(file):
    line = json.loads(line)
    texts = line['text']
    features = tokenizer.batch_encode_plus(texts, padding=True, return_token_type_ids=False, return_tensors='pt')
    labels = [0] * len(texts)
    for i in line['label']:
        labels[i] = 1
    features['labels'] = torch.FloatTensor(labels)
    data.append(features)

data = data[:10000]
file = '/home/vocust001/xly/ccc/extract_dev.json'

test_data = []
for line in open(file):
    line = json.loads(line)
    texts = line['text']
    features = tokenizer.batch_encode_plus(texts, padding=True, return_token_type_ids=False, return_tensors='pt')
    labels = [0] * len(texts)
    for i in line['label']:
        labels[i] = 1
    features['labels'] = torch.FloatTensor(labels)
    test_data.append(features)


class TaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = get_bert_model('roberta')
        self.cnn = ResidualGatedConv()
        self.dense = Dense(1, activation='sigmoid')

    def forward(self, batch):
        inputs_ids = batch['input_ids']
        mask = batch['attention_mask']
        # logtis = self.bert(inputs_ids, mask)[1].unsqueeze(0).detach()
        logtis = self.bert(inputs_ids, mask)[1]

        prob = self.dense(logtis)
        prob = prob.unsqueeze(0)

        # prob = self.cnn(logtis)
        return prob


# class TaskModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.bert = get_bert_model('roberta')
#         self.rnn = nn.LSTM(768, 300, batch_first=True)
#         self.dense = Dense(1, activation='sigmoid')
#
#     def forward(self, batch):
#         inputs_ids = batch['input_ids']
#         mask = batch['attention_mask']
#         logtis = self.bert(inputs_ids, mask)[1].unsqueeze(0).detach()
#         # logtis = self.bert(inputs_ids, mask)[1].unsqueeze(0)
#
#         # prob = self.cnn(logtis)
#         logtis = self.rnn(logtis)[0]
#         prob = self.dense(logtis)
#         return prob


device = 'cuda'
model = TaskModel()
model.to(device)
adam = torch.optim.Adam(model.parameters(), lr=1e-5)
test_num = 1000

train, test = data[:-test_num], data[-test_num:]
if 0:
    best_recall = 0
    for _ in range(3):
        model.train()
        for i in tqdm.tqdm(range(len(data))):
            cur = data[i]
            cur = {k: v.to(device) for k, v in cur.items()}
            # prob = model(cur).squeeze(0)
            prob = model(cur)
            # print(prob.shape)
            prob = prob.squeeze(0)
            loss_fn = nn.BCELoss()
            loss = loss_fn(prob, cur['labels'].unsqueeze(1))
            # print(round(loss.item(), 4))
            loss.backward()
            adam.step()
            adam.zero_grad()

        model.eval()

        threshold = [0.08,  0.1,  0.2, 0.3,  0.4, 0.5, 0.6,0.7]
        yt, yp = [], [[] for i in range(len(threshold))]
        from sklearn.metrics import classification_report

        for i in range(len(test_data)):
            cur = test_data[i]
            cur = {k: v.to(device) for k, v in cur.items()}
            with torch.no_grad():
                prob = model(cur).squeeze(0).squeeze(-1)
                # prob = model(cur).squeeze(-1)

            yt.extend(cur['labels'].cpu().numpy().tolist())
            for idx, j in enumerate(threshold):
                yp[idx].extend((prob > j).long().cpu().numpy().tolist())
        for i in yp:
            print(sum(i))
            print(classification_report(yt, i))

        yt = np.array(yt)
        yp = np.array(yp[1])
        tp = (yt == 1) & (yp == 1)
        recall = tp.sum() / yt.sum()
        if recall > best_recall:
            torch.save(model, 'extract_model.bin')
            best_recall = recall

model = torch.load('extract_model.bin')


### 预测
if 1:
    datas = []
    file = '/home/vocust001/xly/ccc/A类预测.json'
    test_data = []
    for line in open(file):
        line = json.loads(line)
        text = line['text']
        datas.append(line)
        features = tokenizer.batch_encode_plus(text, padding=True, return_token_type_ids=False, return_tensors='pt')
        test_data.append(features)

fout = open('/home/vocust001/xly/ccc/A类预测_extracted.json', 'w+')

for i in range(len(test_data)):
    cur = test_data[i]
    cur = {k: v.to(device) for k, v in cur.items()}
    with torch.no_grad():
        prob = model(cur).squeeze(0).squeeze(-1)
        # prob = model(cur).squeeze(-1)
    pred = (prob > 0.1).cpu().numpy()
    select_text = np.array(datas[i]['text'])[pred].tolist()
    datas[i]['text'] = select_text
    # preds.append(pred)
    # yt.extend(cur['labels'].cpu().numpy().tolist())
    # for idx, j in enumerate([0.04, 0.05, 0.06, 0.07, ]):
    #     yp[idx].extend((prob > j).long().cpu().numpy().tolist())
for line in datas:
    fout.write(json.dumps(line, ensure_ascii=False) + '\n')

# fout = open('/home/vocust001/xly/ccc/extract_dev_predict.json', 'w+')
# count = 0
# for line in open(file):
#     not_recall = []
#     line = json.loads(line)
#     line['pred'] = preds[count]
#     assert len(line['pred']) == len(line['text'])
#     for i in line['label']:
#         if preds[count][i] != 1:
#             not_recall.append(line['text'][i])
#
#     line['not_recall'] = not_recall
#     if not_recall:
#         print(line)
#     count += 1
#
#     fout.write(json.dumps(line, ensure_ascii=False) + '\n')


