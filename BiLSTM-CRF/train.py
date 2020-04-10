import  pickle
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import codecs
from model import Model

def calculate(x,y,id2word,id2tag,res=[]):
    entity=[]
    for j in range(len(x)):
        if id2tag[y[j]]=='B':
            entity=[id2word[x[j]]]
        elif id2tag[y[j]]=='M' and len(entity)!=0:
            entity.append(id2word[x[j]])
        elif id2tag[y[j]]=='E' and len(entity)!=0:
            entity.append(id2word[x[j]])
            res.append(entity)
            entity=[]
        elif id2tag[y[j]]=='S':
            entity=[id2word[x[j]]]
            res.append(entity)
            entity=[]
        else:
            entity=[]
    return res


with open('../data/datasave.pkl', 'rb') as inp:
    word2id = pickle.load(inp)
    id2word = pickle.load(inp)
    tag2id = pickle.load(inp)
    id2tag = pickle.load(inp)
    x_train = pickle.load(inp)
    y_train = pickle.load(inp)
    x_test = pickle.load(inp)
    y_test = pickle.load(inp)
START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 100
HIDDEN_DIM = 200
EPOCHS = 5
LR=0.005
tag2id[START_TAG]=len(tag2id)
tag2id[STOP_TAG]=len(tag2id)

model = Model(len(word2id) + 1, tag2id, EMBEDDING_DIM, HIDDEN_DIM)

optimizer = optim.SGD(model.parameters(), lr=LR, weight_decay=1e-4)



for epoch in range(EPOCHS):
    index = 0
    for sentence, tags in zip(x_train, y_train):
        index += 1
        model.zero_grad()

        sentence = torch.tensor(sentence, dtype=torch.long)
        tags = torch.tensor(tags, dtype=torch.long)

        loss = model(sentence, tags)

        loss.backward()
        optimizer.step()
        if index % 10000 == 0:
            print("epoch", epoch, "index", index)
    entityres = []
    entityall = []
    for sentence, tags in zip(x_test, y_test):
        sentence = torch.tensor(sentence, dtype=torch.long)
        score, predict = model.test(sentence)
        entityres = calculate(sentence, predict, id2word, id2tag, entityres)
        entityall = calculate(sentence, tags, id2word, id2tag, entityall)

    rightpre = [i for i in entityres if i in entityall]
    if len(rightpre) != 0:
        precision = float(len(rightpre)) / len(entityres)
        recall = float(len(rightpre)) / len(entityall)
        print("precision: ", precision)
        print("recall: ", recall)
        print("fscore: ", (2 * precision * recall) / (precision + recall))
    else:
        print("precision: ", 0)
        print("recall: ", 0)
        print("fscore: ", 0)

    path_name = "./model/model" + str(epoch) + ".pkl"
    torch.save(model, path_name)
    print("model has been saved in  ", path_name)


