import argparse
from data_preprocess import Trie
from tqdm import tqdm
import re


parser = argparse.ArgumentParser()
parser.add_argument('--corpus', required=True)
parser.add_argument('--embedding', required=True)
args = parser.parse_args()


def shrink(corpus, embedding):
    vocabs = []
    lines = {}
    print('Building Vocab')
    for i, line in enumerate(open(embedding, encoding="utf-8")):
        if i == 0 and line.count(' ') < 10:
            continue
        vocab = line.split()[0]
        lines[vocab] = i
        if len(vocab) > 1 and re.search('[^\x00-\xff]', vocab):
            vocabs.append(vocab)
    print('Building trie')
    trie = Trie(vocabs)
    keep_lines = set()
    corpus = open(corpus, encoding="utf-8")
    for sent in tqdm(corpus):
        for word in trie.get_lexicon(sent.rstrip()):
            word = word[2]
            keep_lines.add(lines[word])

    print('Writing new file to {}'.format(embedding[:-4] + 'shrink.txt'))
    fout = open(embedding[:-4] + 'shrink.txt', 'w+', encoding="utf-8")
    for i, line in enumerate(open(embedding, encoding="utf-8")):
        if i in keep_lines:
            fout.write(line)
    fout.close()


if __name__ == "__main__":
    shrink(args.corpus, args.embedding)
