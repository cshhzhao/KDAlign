from __future__ import division
from __future__ import print_function

import pickle as pk
import itertools

import random
import time
import argparse
import os
import urllib.request
import json

import torch
import torch.multiprocessing
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

# from tree_rule_process.model.pygcn.pygcn.utils import load_data, accuracy, RunningAvg, Mydataset
# from tree_rule_process.model.pygcn.pygcn.models import GCN, MLP
from utils import load_data, accuracy, RunningAvg, Mydataset
from models import GCN, MLP
#-------------------------------------------------------------------------
#只使用cpu进行计算
#不使用gpu计算
def main():
    torch.multiprocessing.set_sharing_strategy('file_system')

    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001,     #人工调整学习率~
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-3,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=64,
                        help='Number of hidden units.') #和GAT一致 注意
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--indep_weights', action="store_true", default=False,
                        help='whether to use independent weights for different types of gates in ddnnf')
    parser.add_argument('--dataset', type=str, required=True,
                        help='dataset')
    parser.add_argument('--ds_path', type=str, required=True)
    parser.add_argument('--w_reg', type=float, default=0.1, help='strength of regularization')
    parser.add_argument('--cls_reg', type=float, default=0.1, help='weight of classification loss')
    parser.add_argument('--directed', action='store_true', default=False)
    parser.add_argument('--margin', type=float, default=1.0, help='margin in triplet margin loss')
    parser.add_argument('--dataloader_workers', type=int, default=0, help='number of workers for dataloaders')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for training')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    for arg in vars(args):
        print(f'{arg:>30s} = {str(getattr(args, arg)):<30s}')

    ds_path = args.ds_path

    indep_weights = args.indep_weights
    dataset = args.dataset

    reg_name = '.reg' + str(args.w_reg)
    indepname = '.ind' if indep_weights else ''
    directed_name = '.dir' if args.directed else ''
    cls_reg_name = '.cls' + str(args.cls_reg)
    seed_name = '.seed' + str(args.seed)

    # Load data
    and_or = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def cpu_input(adj, features, labels, idx_train, idx_val, idx_test):
        features = features
        adj = adj
        labels = labels
        idx_train = idx_train
        idx_val = idx_val
        idx_test = idx_test
        return adj, features, labels, idx_train, idx_val, idx_test

    # to add regularization for AND OR gate
    def andloss(children):
        temp = torch.mm(torch.t(children), children)
        loss = torch.sum(torch.abs(temp - torch.diag(temp)))
        return loss

    def orloss(children):
        loss = (torch.norm(torch.sum(children, dim=0).squeeze(0)) - 1) ** 2
        return loss

    # file_name_debug = []
    # data_all = []
    # data_train = []
    # data_test = []
    # if dataset in ['ddnnf', 'cnf', 'general', 'vrd']:
    file_list_raw = os.listdir(f'{ds_path}/{dataset}/')
    file_list = set()
    for file in file_list_raw:
        if '.s' not in file and '.and' not in file and '.or' not in file and '.rel' not in file:
            file_list.add(urllib.request.unquote(file))
    file_list = list(file_list)

    # split train test
    # shuffle first
    idx = np.arange(len(file_list))
    np.random.seed(1)
    np.random.shuffle(idx)
    file_list = np.array(file_list)[idx]

    split_idx = int(round(len(file_list) * 0.9, 0))
    # file_list_train = file_list[:split_idx]
    # file_list_test = file_list[split_idx:]
    file_list_train = file_list
    file_list_test = file_list
    json.dump(list(file_list_test), open(dataset + seed_name+'_no_val' + '.testformula', 'w'), ensure_ascii=False)

    print('file list length: ', len(file_list), len(file_list_train), len(file_list_test))

    #为了更全面的训练embeder，训练集是全集
    # dataset_train = Mydataset(dataset, file_list_train, and_or=and_or, ds_path=ds_path, args=args)
    dataset_train = Mydataset(dataset, file_list, and_or=and_or, ds_path=ds_path, args=args)
    dataset_test = Mydataset(dataset, file_list, and_or=and_or, ds_path=ds_path, args=args)

    #dataloader_train里面是空的内容 注意！！！！！
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.dataloader_workers)
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=args.dataloader_workers)

    print('Number of training samples:', len(dataset_train))

    # Model and optimizer
    # a dummy example to determine the dimension of input data
    adj0, features0, labels0, idx_train0, idx_val0, idx_test0, add_children0, or_children0 = load_data(
        '0', dataset, and_or=and_or, override_path=ds_path)

    model = GCN(nfeat=features0.shape[1],
                nhid=args.hidden,
                # nclass=labels.max().item() + 1,
                nclass=args.hidden,   #和GAT对应
                dropout=args.dropout,
                indep_weights=indep_weights)
    # def __init__(self, ninput=89 * 2, nhidden=100, nclass=2, dropout=0):
    mlp = MLP(dropout=args.dropout,ninput=args.hidden*2)
    optimizer = optim.Adam(itertools.chain(model.parameters(), mlp.parameters()),
                           lr=args.lr, weight_decay=args.weight_decay)

    creterion = torch.nn.TripletMarginLoss(margin=args.margin, p=2)
    CE = torch.nn.CrossEntropyLoss()

    loss_list = RunningAvg(window_size=200)
    loss_list_CE = RunningAvg(window_size=200)
    acc_list = RunningAvg(window_size=200)
    loss_by_iter = []

    if args.cuda:
        model.cuda()
        mlp.cuda()

    def train_step(epoch, loss_save):

        count_machine = 0  # 用来计数，目的是每100张图片查看一下当前的效果
        test_start_time = time.time() #查看的是累计的时间
        for _, data_train_group in tqdm(enumerate(dataloader_train), desc='Training', total=len(dataloader_train)):
            # pass three times first, then back propagate the loss
            model.train() #模型启用dropout的意思
            mlp.train()

            count_machine+=1

            for data_train in data_train_group:
                vector3 = []
                regularization = 0
                for data_train_item in data_train:
                    # and/or children: [[1,2],[3,4]]
                    adj, features, labels, idx_train, idx_val, idx_test = cpu_input(*data_train_item[:-2]) #这里是对载入的数据转为cuda对象
                    #注意labels代表的是logic graph中一个节点是global、leaf node、and、or还是not节点
                    and_children, or_children = data_train_item[-2:] #得到里面的and节点和children节点
                    t = time.time()  #开始时间
                    output = model(features.squeeze(0), adj.squeeze(0), labels.squeeze(0)) #将当前数据送入GCN模型得到输出
                    vector3.append(output[0])
                    # add regularization
                    if and_or:
                        if len(and_children) != 0:
                            for addgate in range(len(and_children)):
                                add_child_tensor = None
                                for childidx in range(len(and_children[addgate])):
                                    if add_child_tensor is None:
                                        add_child_tensor = output[and_children[addgate][childidx]]
                                    else:
                                        add_child_tensor = torch.cat(
                                            (add_child_tensor, output[and_children[addgate][childidx]]))
                                regularization += andloss(add_child_tensor)
                        if len(or_children) != 0:
                            for orgate in range(len(or_children)):
                                or_child_tensor = None
                                for childidx in range(len(or_children[orgate])):
                                    if or_child_tensor is None:
                                        or_child_tensor = output[or_children[orgate][childidx]]
                                    else:
                                        or_child_tensor = torch.cat(
                                            (or_child_tensor, output[or_children[orgate][childidx]]))
                                regularization += orloss(or_child_tensor)

                # back prop GCN
                loss_train = creterion(vector3[0].unsqueeze(0), vector3[1].unsqueeze(0), vector3[2].unsqueeze(0))  #这是在计算triplet loss                                                                                               #vector3向量三个维度分别是F,st,sf；规则，满足，不满足
                loss_train += args.w_reg * regularization
                loss_list.add(float(loss_train))  #.cpu()的目的是loss_train本身是在cuda中计算的，取出数据到cpu中需要这个函数转换
                optimizer.zero_grad()
                # loss_train.backward()

                # back prop MLP
                mlp.train()
                input = torch.cat((torch.cat((vector3[0], vector3[1])).unsqueeze(0), torch.cat((vector3[0], vector3[2])).unsqueeze(0)))
                pred = mlp(input)
                target = torch.LongTensor([1, 0])
                mlp_loss = CE(pred, target)
                loss_by_iter.append(float(mlp_loss))
                (loss_train + args.cls_reg * mlp_loss).backward()
                optimizer.step()
                loss_list_CE.add(float(mlp_loss))

                # calculate accuracy
                _, predicted = torch.max(pred.data, 1)
                acc = (predicted == target).sum().item() / target.size(0)
                acc_list.add(float(acc))

            if(count_machine%100==0):#每100张图片输出一次当前的效果，效果的loss等是累加的注意！！！！
                test_end_time=time.time()
                loss_avg = loss_list.avg()
                CE_loss_avg = loss_list_CE.avg()
                acc_avg = acc_list.avg()
                print('Epoch: {:04d} '.format(epoch + 1),
                      'Graph Number:{} '.format(count_machine),
                      'Avg loss: {:.4f} '.format(loss_avg),
                      'Avg CE loss: {:.4f} '.format(CE_loss_avg),
                      'Avg Acc: {:.4f} '.format(acc_avg),
                      'time: {:.4f}s '.format(test_end_time-test_start_time)) #时间是累加时间比如 0-100,0-200,0-300，。。。的累加时间

        loss_avg = loss_list.avg()
        CE_loss_avg = loss_list_CE.avg()
        acc_avg = acc_list.avg()

        #这里输出的是一轮训练需要的总时间以及此时的loss
        print('Epoch: {:04d}'.format(epoch + 1),
              'Avg loss: {:.4f}'.format(loss_avg),
              'Avg CE loss: {:.4f}'.format(CE_loss_avg),
              'Avg Acc: {:.4f}'.format(acc_avg),
              'time: {:.4f}s'.format(time.time() - t))
        loss_save['triplet_loss'].append(loss_avg)
        loss_save['CE_loss'].append(CE_loss_avg)
        loss_save['acc'].append(acc_avg)

        return loss_avg, CE_loss_avg, acc_avg

    def test():
        avg_loss = []
        avg_loss_CE = []
        total = 0
        correct = 0
        model.eval()
        mlp.eval()
        for _, data_test_group in tqdm(enumerate(dataloader_test), desc='Testing', total=len(dataloader_test)):
            # pass three times first, then back propagate the loss
            for data_test in data_test_group:
                vector3 = []
                for data_test_item in data_test:
                    adj, features, labels, idx_train, idx_val, idx_test = cpu_input(*data_test_item[:-2])
                    output = model(features.squeeze(0), adj.squeeze(0), labels.squeeze(0))
                    vector3.append(output[0])

                loss_test = creterion(vector3[0].unsqueeze(0), vector3[1].unsqueeze(0), vector3[2].unsqueeze(0))
                avg_loss.append(float(loss_test))

                # back prop MLP
                input = torch.cat(
                    (
                    torch.cat((vector3[0], vector3[1])).unsqueeze(0), torch.cat((vector3[0], vector3[2])).unsqueeze(0)))
                pred = mlp(input)
                target = torch.LongTensor([1, 0])
                mlp_loss = CE(pred, target)
                avg_loss_CE.append(float(mlp_loss))

                # caclulate accuracy
                _, predicted = torch.max(pred.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        avg_loss = np.average(avg_loss)
        avg_loss_CE = np.average(avg_loss_CE)
        avg_acc = correct / total
        print('Test loss: {:.4f}'.format(avg_loss),
              'Test loss CE: {:.4f}'.format(avg_loss_CE),
              'Test Acc: {:.4f}'.format(avg_acc), )
        return avg_loss, avg_loss_CE, avg_acc

    # save reg in the name

    # Train model
    train_loss_save = {'triplet_loss': [], 'CE_loss': [], 'acc': []}
    # test_loss_save = {'triplet_loss': [], 'CE_loss': [], 'acc': []}
    t_total = time.time()
    best_acc = 0.0
    for epoch in range(args.epochs):


        train_status = train_step(epoch, train_loss_save)

        if train_status[2] > best_acc: #自动保存效果最好的embeder
            best_acc = train_status[2]
            torch.save(model,'./model_save/' + dataset + reg_name + indepname + directed_name + cls_reg_name + seed_name +'_no_val'+ '.model')
            torch.save(mlp,'./model_save/' + dataset + reg_name + indepname + directed_name + cls_reg_name + seed_name +'_no_val'+ '.mlp.model')
            print('\tNew best model saved.')
        json.dump(train_loss_save, open('./acc_loss/' + dataset + reg_name + indepname + directed_name + cls_reg_name + seed_name+'_no_val' + '.train_save','w'),ensure_ascii=False)
        pk.dump(loss_by_iter, open('./acc_loss/' + dataset + reg_name + indepname + directed_name + cls_reg_name + seed_name+'_no_val' + '.loss_by_iter','wb'))

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    print(f"Best Train acc: {max(train_loss_save['acc'])}, at epoch: {np.argmax(train_loss_save['acc'])}")

if __name__=='__main__':
    main()