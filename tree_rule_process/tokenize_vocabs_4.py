import sys

sys.path.append('.')
sys.path.append('../')

import pickle as pk
import json
import pdb

from tree_rule_process.model.misc.Glove import load_glove_as_dict

if __name__=='__main__':

    target_dataset='Cardiotocography'

    glove = load_glove_as_dict('./tree_rule_process/glove', 50, identifier='6B') #加载的用Glove训练的word_embedding是50维的
    objs = pk.load(open('./tree_rule_process/'+target_dataset+'/objects.pk','rb'))
    #注意除了pres要转换 objs也需要转换！！！！！！！
    pres = pk.load(open('./tree_rule_process/'+target_dataset+'/predicates.pk','rb'))

    tokenizer={'token2vocab':{},'vocab2token':{}} #开始是空的，然后不断地遍历，往这个字典里添加内容
                                                #字典的内容其实是建立一个和word_vector中embedding的索引
                                                #比如 第1个获取的单词是person，那么此时注意：
                                                #tokenizer = {'token2vocab': {'person':1}, 'vocab2token': {1:'person'}}
                                                #word_vector[[glove(person],]
                                                #相当于是通过在tokenizer中查找object的word vector的id，然后找到这个词语的embedding
                                                #这样可以避免每次构造logic graph的时候都需要调用glove获取embedding
                                                #属于预处理操作
    current_token=0 #当前已经处理了多少个object和predicate
    word_vector=[]

    def add_token(tokenizer, object, current_token, word_vector, glove):
        tokenizer['vocab2token'][object] = current_token
        tokenizer['token2vocab'][current_token] = object
        current_token += 1
        try:
            word_vector.append(glove[object.lower()])
        except Exception as e:
            print(object)
            word_vector.append(glove['<unk>'])
        return tokenizer, current_token

    #这里是处理句子，不需要考虑，objects保存的都是全称，只有在前面find_rels的处理告警信息的时候才需要做全称对应，后面保存如var_pool等里面都是全称了。
    for obj in objs:  #开始遍历

        if len(obj.split()) == 1:
            tokenizer, current_token = add_token(tokenizer, obj.split()[0], current_token, word_vector,
                                                glove) #obj.split()[0]才是要真正的输入，tokenizer、current_token、word_vector都是记录内容
                                                        #glove是已经训练好的用于word_embedding的Glove模型

        elif len(obj.split()) > 1: #如果object的名称超过一个单词，那么一个一个的拆分即可
            for w in obj.split():
                if w not in tokenizer['vocab2token'].keys():
                    tokenizer, current_token = add_token(tokenizer, w, current_token, word_vector,
                                                        glove)

    for pre_1 in pres:
        if len(pre_1.split()) == 1:
            tokenizer, current_token = add_token(tokenizer, pre_1.split()[0], current_token,word_vector, glove)
        elif len(pre_1.split()) > 1:  # 这里面向的是诸如in the front of等这种词语的构造
            for w in pre_1.split():  # 比如in the front of 每个单词都需要得到其的glove的word embedding
                if w not in tokenizer['vocab2token'].keys():  # 看是否这个单词已经存在
                    tokenizer, current_token = add_token(tokenizer, w, current_token, word_vector,glove)

    pk.dump(tokenizer, open('./tree_rule_process/'+target_dataset+'/tokenizers.pk','wb')) # word vectors的索引
    pk.dump(word_vector, open('./tree_rule_process/'+target_dataset+'/word_vectors.pk', 'wb'))  # 根据id的自然顺序存放了不同单词的embedding，glove类型


#
# if __name__=='__main__':
#     f=open('../data/tokenizers.pk','rb')
#     tok=pk.load(f)
#     print(tok)

