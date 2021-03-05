import json
import pandas as pd

file = ['./train.json', './dev.json']

# corpus = []


def convert(f):
    lines = []
    for line in open(f, encoding='utf-8'):
        line = json.loads(line)
        line2 = [line['text']]
        corpus.append(line['text'])

        labels = []
        for cate, entity_dict in line['label'].items():
            for entity, span, in entity_dict.items():
                sub_line = {'entity': entity, 'label': cate, 'spans': span}
                labels.append(sub_line)
        line2.append(labels)
        print(line2)
        lines.append(line2)

    df = pd.DataFrame(lines)
    print(df.head())
    df.to_csv(f.replace('json', 'csv'), index=None)


convert(file[0])
convert(file[1])
# pd.DataFrame(corpus).to_csv('clue_ner_corpus.txt', index=None, header=None)
