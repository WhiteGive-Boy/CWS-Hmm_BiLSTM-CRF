import  pickle
Trans = {}  #trans
Emit = {}  #emit
Count_dic = {}
Start = {}  #start

with open('../data/datasave.pkl', 'rb') as inp:
    '''
    读取数据处理结果
    '''
    word2id = pickle.load(inp)
    id2word = pickle.load(inp)
    tag2id = pickle.load(inp)
    id2tag = pickle.load(inp)
    x_train = pickle.load(inp)
    y_train = pickle.load(inp)
    x_test = pickle.load(inp)
    y_test = pickle.load(inp)


def calculate(x,y,id2word,id2tag,res=[]):
    '''

    :param x: 输入的句子(转换后的ID序列）
    :param y: 标注tag序列
    :param id2word: id2word
    :param id2tag: id2tag
    :param res: 添加输入句子的词组划分 BME S
    :return: res
    '''
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




def init():

    for tag in tag2id:
        Trans[tag2id[tag]] = {}
        for tag2 in tag2id:
            Trans[tag2id[tag]][tag2id[tag2]] = 0.0
    for tag in tag2id:
        Start[tag2id[tag]] = 0.0
        Emit[tag2id[tag]] = {}
        Count_dic[tag2id[tag]] = 0
def train():
    '''
    根据输入的训练集进行各个数组的填充
    :return:
    '''
    for sentence, tags in zip(x_train, y_train):
        for i in range(len(tags)):
            if i == 0:
                Start[tags[0]] += 1
                Count_dic[tags[0]] += 1
            else:
                Trans[tags[i - 1]][tags[i]] += 1
                Count_dic[tags[i]] += 1
                if sentence[i] not in Emit[tags[i]] :
                    Emit[tags[i]][sentence[i]] = 0.0
                else:
                    Emit[tags[i]][sentence[i]] += 1

    for tag in Start:
        Start[tag] = Start[tag] * 1.0 / len(x_train)
    for tag in Trans:
        for tag1 in Trans[tag]:
            Trans[tag][tag1] = Trans[tag][tag1] / Count_dic[tag]

    for tag in Emit:
        for word in Emit[tag]:
            Emit[tag][word] = Emit[tag][word] / Count_dic[tag]
    print(Start)
    print(Trans)

def viterbi(sentence, tag_list):
    '''

    :param sentence:  输入的句子
    :param tag_list:  所有的tag
    :return: prob预测的最大的概率 bestpath 预测的tag序列
    '''
    V = [{}] #tabular
    path = {}
    backpointers = []
    for y in tag_list: #init
        V[0][y] = Start[y] * (Emit[y].get(sentence[0],0.00000001))
        path[y]=y
    backpointers.append(path)
    for t in range(1,len(sentence)):
        V.append({})
        newpath = {}
        path = {}
        for y in tag_list:
            (prob,state ) = max([(V[t-1][y0] * Trans[y0].get(y,0.00000001) * Emit[y].get(sentence[t],0.00000001) ,y0) for y0 in tag_list])
            V[t][y] =prob
            path[y]=state
        backpointers.append(path)
    (prob, state) = max([(V[len(sentence) - 1][y], y) for y in tag_list])
    best_path=[]
    best_path.append(state)
    for pathi in reversed(backpointers):
        state = pathi[state]
        best_path.append(state)
    best_path.pop()
    # Pop off the start tag (we dont want to return that to the caller)
    best_path.reverse()
    return (prob, best_path)

def test():
    '''
    计算Precision和Recall以及Fscore
    '''
    taglist=[tag2id[tag] for tag in tag2id]
    entityres = []#根据预测结果的分词序列
    entityall = []#根据真实结果的分词序列
    for sentence, tags in zip(x_test, y_test):
        #score, predict=viterbi(sentence,taglist,Start,Trans,Emit)
        score, predict = viterbi(sentence, taglist)
        entityres = calculate(sentence, predict, id2word, id2tag, entityres)
        entityall = calculate(sentence, tags, id2word, id2tag, entityall)

    rightpre = [i for i in entityres if i in entityall]#预测成功的分词序列
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


if __name__ == "__main__":
    init()
    train()
    test()