from tqdm import tqdm, trange
import torch
import numpy as np
import collections
from utils import *
from torchcontrib.optim import SWA


class Model(object):
    def __init__(self, model, optimizer, device='cuda', gradient_accumulation_steps=1, max_grad_norm=None):
        model.to(device)
        self.model = model
        self.device = device
        self.optimizer = optimizer

        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm

    def get_merics(self, name):
        if name == 'ner':
            return self.evaluate_ner
        if name == 'biaffine':
            return self.evaluate_biaffine

    def fit(self, train_dataloader, epochs=1, dev_dataloader=None, metrics=None, apply_fgm=False):
        if dev_dataloader is not None:
            metric_raw = 0
        for e in trange(1, epochs + 1, desc='Epoch'):
            if apply_fgm:
                fgm = FGM(self.model)

            self.model.train()
            iter_bar = tqdm(train_dataloader, desc='Iter (loss=X.XXX)')
            losses = []
            for step, batch in enumerate(iter_bar):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                logits = self.model(batch)
                loss = self.model.compute_loss(logits, batch)
                losses.append(loss.item())
                iter_bar.set_description('Iter (loss=%5.3f)' % np.mean(losses))
                if self.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps
                loss.backward(retain_graph=apply_fgm)
                if self.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm)
                if apply_fgm:
                    fgm.attack(emb_name='word_embeddings')  # 在embedding上添加对抗扰动
                    logiths = self.model(batch)
                    loss_adv = self.model.compute_loss(logits, batch)
                    loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                    fgm.restore(emb_name='word_embeddings')  # 恢复embedding参数
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            if dev_dataloader is not None and metrics is not None:
                name, args = metrics
                func = self.get_merics(name)

                metric = func(dev_dataloader, *args)
                if metric > metric_raw:
                    torch.save(self.model, 'best_model')
                    metric_raw = metric
                print("best_metric", metric)

    def forward_batch(self, batch):
        with torch.no_grad():
            return self.model(batch)

    def evaluate_ner(self, data_loader, texts, offset, id2label, y_true):
        y_pred = []
        for batch in data_loader:
            batch = self.move_data_to_device(batch)
            logits = self.forward_batch(batch)
            y_pred.extend(self.model.output.crf.decode(logits, batch['attention_mask']))
        y_pred = extract_texts(y_pred, texts, offset, id2label)

        f1 = evaluate_span_f1(y_pred, y_true)
        return f1

    def evaluate_biaffine(self, data_loader, texts, offset, id2label, y_true):
        y_pred = []
        for batch in data_loader:
            batch = self.move_data_to_device(batch)
            logits = self.forward_batch(batch)
            for l in logits:
                y_pred.append(l)

        y_pred = extract_biaffine(y_pred, texts, offset, id2label)
        f1 = evaluate_span_f1(y_pred, y_true)
        return f1

    def move_data_to_device(self, batch):
        return {k: v.to(self.device) for k, v in batch.items()}

    def predict(self, dataloader, mask_key=None, return_type='tensor'):

        self.model.eval()
        res = []

        # 根据mask输出实际长度
        if mask_key is not None:
            for batch in tqdm(dataloader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                with torch.no_grad():
                    logits = self.model(batch)
                    for logit, mask in zip(logits, batch[mask_key]):
                        logit = logit[mask.bool()]
                        if self.device[:4] == 'cuda':
                            logit = logit.cpu()
                        res.append(logit.numpy())
            return res

        else:
            for batch in tqdm(dataloader, desc='predict'):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                with torch.no_grad():
                    logits = self.model(batch)
                res.append(logits)
            res = torch.cat(res, 0)

            if return_type == 'numpy':
                if 'cuda' in self.device:
                    res = res.cpu()
                return res.numpy()
            return res


def extract_biaffine(labels, texts, offsets, id2label):
    output = []
    for text, offset, label in zip(texts, offsets, labels):
        line = {}
        seq_len = len(offset)
        indices_mat = torch.triu(torch.ones((seq_len, seq_len)), diagonal=0, out=None).to(labels[0].device)
        indices = torch.where(indices_mat > 0)
        logits = label[indices[0], indices[1]].argmax(-1)

        label_mapping = {}
        count = 0
        for i in range(seq_len):
            for j in range(i, seq_len):
                label_mapping[count] = (i, j)
                count += 1

        for idx, value in enumerate(logits):
            if value > 0:
                start, end = label_mapping[idx]
                mapping_start = offset[start][0]
                mapping_end = offset[end][1]
                line[text[mapping_start:mapping_end]] = (id2label[value - 1], mapping_start, mapping_end - 1)

        output.append(line)
    return output


def evaluate_span_f1(y_pred, y_true, ignore_spans=True):
    if ignore_spans:
        y_pred = [{(x['entity'], x['label']) for x in y} for y in y_pred]
        y_true = [{(x['entity'], x['label']) for x in y} for y in y_true]

    tp = fp = fn = 0
    for y_p, y_t in zip(y_pred, y_true):
        tp_pair = y_p & y_t
        fp_pair = y_p - tp_pair
        fn_pair = y_t - tp_pair
        tp += len(tp_pair)
        fp += len(fp_pair)
        fn += len(fn_pair)
    f1 = 2 * tp / (2 * tp + fp + fn)
    return f1


def extract_texts(labels, texts, offsets, id2label):
    start = False
    output = []
    for text, offset, label in zip(texts, offsets, labels):
        line = {}
        entity = collections.defaultdict(list)
        seq_len = len(offset)
        label = label[:seq_len]
        for i, v in enumerate(label):
            if v % 2 == 1:
                label = id2label[(v - 1) // 2]
                entity[label].append([i])
                last_start = label
                start = True
            elif start and v != 0:
                label = id2label[(v - 1) // 2]
                if label == last_start:
                    entity[label][-1].append(i)
            else:
                start = False

        for k, v in entity.items():
            if len(v) > 0:
                for j in v:
                    start = j[0]
                    end = j[-1]
                    mapping_start = offset[start][0]
                    mapping_end = offset[end][1]
                    entity_text = text[mapping_start:mapping_end]
                    if entity_text not in line:
                        line[entity_text] = {"entity": entity_text, 'label': k,
                                             'spans': [(mapping_start, mapping_end - 1)]}
                    else:
                        line[entity_text]['spans'].append((mapping_start, mapping_end - 1))
        output.append(list(line.values()))

    return output


class PGD():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., alpha=0.3, emb_name='emb.', is_first_attack=False):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r


class FGM(object):
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='emb.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='emb.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
