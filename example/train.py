from transformers.modeling_bert import BertForPreTraining
from xtools import *
from bert4torch.loss import CrossEntropyLoss
import random
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.distributed as dist


local_rank = 0
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
dist.init_process_group(backend='nccl')

dist.barrier()



# torch.distributed.init_process_group(backend="nccl")

batch_size = 128
model_path = 'hfl/chinese-roberta-wwm-ext'
adam_epsilon = 1e-8
lr = 2e-5
device = 'cuda'

debug = 1

def get_data(path):
    ret = []
    _, name = os.path.split(path)
    name = name.split('-')[0]
    for idx, line in enumerate(tqdm.tqdm(open(path))):
        line = json.loads(line)
        line['task'] = name
        if debug and idx > 100:
            break
    return ret


seq2seq_data = '/home/vocust001/xly/ccc/seq2seq_data.json'
lm_data = '/home/vocust001/xly/ccc/lm_data.json'
mlm_data = '/home/vocust001/xly/ccc/mlm_data.json'

seq2seq_data = get_data(seq2seq_data)
lm_data = get_data(lm_data)
mlm_data = get_data(mlm_data)

seq2seq_data = DataLoader(KeyDataset(seq2seq_data), batch_size=batch_size)
lm_data = DataLoader(KeyDataset(lm_data), batch_size=batch_size)
mlm_data = DataLoader(KeyDataset(mlm_data), batch_size=batch_size)

print('finish loading data')


# class KeyDataset(Dataset):
#     def __init__(self, dict_data):
#         self.data = {'mlm': mlm_data, 'lm': lm_data, 'seq2seq': seq2seq_data}
#         self.n_mlm = len(mlm_data)
#         self.n_lm = len(lm_data)
#         self.n_seq = len(seq2seq_data)
#
#     def __len__(self):
#         return sum(len(x) for x in self.data.values())
#
#     def __getitem__(self, index):


def create_lm_mask(attention_mask, direction='l2r'):
    seq_len = attention_mask.size(-1)
    if attention_mask.nid == 2:
        attention_mask = attention_mask.view(-1, 1, 1, seq_len)

    idxs = torch.range(0, seq_len - 1).to(attention_mask)
    if direction == 'l2r':
        triu = (idxs.unsqueeze(-1) >= idxs).float()
    elif direction == 'r2l':
        triu = (idxs.unsqueeze(-1) <= idxs).float()

    attention_mask = (attention_mask + triu > 1).float()
    return attention_mask


def create_unilm_mask(s):
    idxs = torch.cumsum(s, axis=1)
    mask = idxs[:, None, :] <= idxs[:, :, None]
    mask = mask.float()
    return mask


class UnilmForPreTraining(BertForPreTraining):
    def __init__(self, config):
        super().__init__(config)
        self.loss_fn = CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, token_type_ids, *arg, **kwargs):
        prediction_scores, seq_relationship_score = super().forward(input_ids, attention_mask, token_type_ids)
        return prediction_scores, seq_relationship_score

    def prepare_data_for_pretraining(self, batch, task, use_mlm=False):
        new_batch = batch
        if task == 'mlm':
            new_batch['input_ids'] = new_batch.pop('masked_input_ids')

        elif task == 'lm':
            new_batch['label_mask'] = new_batch.pop('attention_mask')
            if random.random() < 0.5:
                direction = 'l2r'
            else:
                direction = 'r2l'
            new_batch['attention_mask'] = create_lm_mask(new_batch['label_mask'], direction)

        elif task == 'seq2seq':
            new_batch['attention_mask'] = create_unilm_mask(new_batch.pop('attention_mask'))
        return new_batch


## 训练
from torch.nn.parallel import DistributedDataParallel as DDP
from apex import amp

model = UnilmForPreTraining.from_pretrained(model_path)
print('model loaded')
model.to(device)
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(
        nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(
        nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=adam_epsilon)



loss_fn = CrossEntropyLoss()
model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)


for batch in mlm_data:
    batch = model.prepare_data_for_pretraining(batch, 'mlm')
    batch = {k:v.to(device) for k, v in list(batch.items())}

    x = model(**batch)
    print(x)