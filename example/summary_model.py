from bert4torch.layer import *
from data_preprocess import *
from xtools import *
from loss import *
from bert4torch.lightmodel import *
from bert4torch.callback import Checkpoint
from bert4torch.evaluate import *
from bert4torch.model import *
import rouge
import pkbar

rouge = rouge.Rouge()

file = '/home/vocust001/xly/ccc/extract_train.json'
max_len = 512

data = []
for line in open(file):
    line = json.loads(line)
    texts = line['text']
    features = tokenizer.encode_plus(''.join(texts[i] for i in line['label']), line['summary'],
                                     return_token_type_ids=True, max_length=512, truncation='only_first')
    # features['attention_mask'] = create_mask(features)

    data.append(features)

data = data
file = '/home/vocust001/xly/ccc/extract_dev.json'

test_text = []
sums = []
for line in open(file):
    line = json.loads(line)
    texts = line['text']
    # features = tokenizer.encode_plus(''.join(texts[i] for i in line['label']), return_tensors='pt',
    #                                  return_token_type_ids=True)
    sums.append(line['summary'])
    # test_data.append(features)
    test_text.append(''.join(texts[i] for i in line['label']))

train = KeyDataset(data)
train = DataLoader(train, batch_size=10, collate_fn=default_collate, shuffle=True)

model = UniLM.from_pretrained('hfl/chinese-roberta-wwm-ext')

device = 'cuda'
model.to(device)
adam = torch.optim.Adam(model.parameters(), lr=2e-5)


def clean_gen(text):
    text = text.split('，')
    ret = [text[0]]
    for i in text[1:]:
        if i == ret[-1]:
            break
        if len(ret) > 2 and i == ret[-2]:
            break
        ret.append(i)
    return '，'.join(ret)


def generate(text, max_length=50):
    max_content_length = max_len - max_length
    feature = tokenizer.encode_plus(text, return_token_type_ids=True, return_tensors='pt',
                                    max_length=max_content_length)
    feature = {k: v.to(device) for k, v in list(feature.items())}
    content_len = len(feature['input_ids'][0])
    max_length = content_len + max_length
    gen = model.generate(max_length=max_length, eos_token_id=tokenizer.sep_token_id, **feature).cpu().numpy()[0]
    gen = gen[content_len:]
    gen = tokenizer.decode(gen, skip_special_tokens=True).replace(' ', '')
    return clean_gen(gen)


def compute_rouge(source, target):
    """计算rouge-1、rouge-2、rouge-l
    """

    source, target = ' '.join(source), ' '.join(target)
    try:
        scores = rouge.get_scores(hyps=source, refs=target)
        return {
            'rouge-1': scores[0]['rouge-1']['f'],
            'rouge-2': scores[0]['rouge-2']['f'],
            'rouge-l': scores[0]['rouge-l']['f'],
        }
    except ValueError:
        return {
            'rouge-1': 0.0,
            'rouge-2': 0.0,
            'rouge-l': 0.0,
        }


def compute_rouges(sources, targets):
    scores = {
        'rouge-1': 0.0,
        'rouge-2': 0.0,
        'rouge-l': 0.0,
    }
    for source, target in zip(sources, targets):
        score = compute_rouge(source, target)
        for k,v in scores.items():
            scores[k] = v + score[k]

    return {k: v / len(targets) for k, v in scores.items()}


## 是否训练
if 0:
    best = 0
    for _ in range(10):
        model.train()
        for cur in tqdm.tqdm(train):
            cur['attention_mask'] = model.create_mask(cur)
            cur = {k: v.to(device) for k, v in cur.items()}
            prob = model(**cur)[0]
            loss = model.compute_loss(prob, cur)
            loss.backward()
            adam.step()
            adam.zero_grad()

        # 测试
        model.eval()
        gens = []
        for (idx, cur) in enumerate(test_text):
            gen = generate(cur, max_length=30)
            # print(gen)
            gens.append(gen)
        scores = compute_rouges(gens, sums)
        print(scores)
        rouge_l = scores['rouge-l']
        if rouge_l > best:
            best = rouge_l
            torch.save(model, 'summary_model')

model = torch.load('summary_model')

#  预测

if 1:
    file = '/home/vocust001/xly/ccc/A类预测_extracted.json'
    ids = []
    test_text = []
    for line in open(file):
        line = json.loads(line)
        texts = line['text']
        # features = tokenizer.encode_plus(''.join(texts[i] for i in line['label']), return_tensors='pt',
        #                                  return_token_type_ids=True)
        # test_data.append(features)
        test_text.append(''.join(texts))
        ids.append(line['uuid'])


gens = []
for (idx, cur) in enumerate(test_text):
    gen = generate(cur, max_length=20)
    gens.append(gen)
    print(gen)

raw = []
file = '/home/vocust001/xly/ccc/A类预测_extracted.json'
for line in open(file):
    line = json.loads(line)
    raw.append(line['text'])

pd.DataFrame({'uuid': ids, '原文': raw, '机器': gens}).to_excel('测试结果.xlsx', index=None)


# ids = torch.LongTensor([[1, 2, 3, 4]])
# ttype = torch.LongTensor([[0, 0, 1, 1]])
# mask = torch.LongTensor([[1, 1, 1, 1]])
# batch = {'input_ids': ids, 'token_type_ids': ttype, 'attention_mask': mask}
# print(model(batch))
#
# from transformers.modeling_bert import PreTrainedModel,
#
# from transformers.modeling_roberta import RobertaForCausalLM
