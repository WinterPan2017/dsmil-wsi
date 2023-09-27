import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms.functional as VF
from torchvision import transforms

import sys, argparse, os, copy, itertools, glob, datetime
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support
from sklearn.datasets import load_svmlight_file
from collections import OrderedDict
import json
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score, roc_curve
from torch.utils.tensorboard import SummaryWriter
import random

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count

def print_log(tstr, f):
    f.write('\n')
    f.write(tstr)
    f.flush()
    print(tstr)

def set_random_seed(seed, deterministic=False):
    """Set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_bag_feats(csv_file_df, shuffle=False): # modify for unified dataset
    feature = np.load(csv_file_df.iloc[0])["features"]
    if shuffle:
        feature = feature[np.random.permutation(len(feature))]
    label = int(csv_file_df.iloc[1])
    return torch.from_numpy(np.array([[label]])).cuda().float(),  torch.from_numpy(feature).cuda().float()

def train(train_df, milnet, criterion, optimizer, args, meters, tb, epoch):
    milnet.train()
    total_loss = 0
    for i in range(len(train_df)):
        optimizer.zero_grad()
        bag_label, bag_feats = get_bag_feats(train_df.iloc[i], shuffle=True)
        ins_prediction, bag_prediction, _, _ = milnet(bag_feats)
        max_prediction, _ = torch.max(ins_prediction, 0, keepdim=True)        
        bag_loss = criterion(bag_prediction, bag_label)
        max_loss = criterion(max_prediction, bag_label)
        loss = 0.5*bag_loss + 0.5*max_loss
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item()
        meters["loss"].update(loss.item())
        meters["bag_loss"].update(bag_loss.item())
        meters["max_loss"].update(max_loss.item())
        step = epoch * len(train_df) + i+1
        tb.add_scalar("train/loss", meters["loss"].avg, step)
        tb.add_scalar("train/bag_loss", meters["bag_loss"].avg, step)
        tb.add_scalar("train/max_loss", meters["max_loss"].avg, step)
        tb.add_scalar("train/lr", optimizer.param_groups[0]['lr'], step)
        sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f' % (i, len(train_df), loss.item()))
    return total_loss / len(train_df)

def test(test_df, milnet, criterion, args): # modified for unified eval metrics
    milnet.eval()
    total_loss = 0
    test_labels = []
    test_predictions = []
    with torch.no_grad():
        for i in range(len(test_df)):
            bag_label, bag_feats = get_bag_feats(test_df.iloc[i])
            ins_prediction, bag_prediction, _, _ = milnet(bag_feats)
            max_prediction, _ = torch.max(ins_prediction, 0, keepdim=True)  
            bag_loss = criterion(bag_prediction, bag_label)
            max_loss = criterion(max_prediction, bag_label)
            loss = 0.5*bag_loss + 0.5*max_loss
            total_loss = total_loss + loss.item()
            sys.stdout.write('\r Testing bag [%d/%d] bag loss: %.4f' % (i, len(test_df), loss.item()))
            test_labels.extend([bag_label[0][0].item()])
            if args.average:
                test_predictions.extend([(0.5*torch.sigmoid(max_prediction)+0.5*torch.sigmoid(bag_prediction)).squeeze().cpu().numpy()])
            else: 
                test_predictions.extend([(0.0*torch.sigmoid(max_prediction)+1.0*torch.sigmoid(bag_prediction)).squeeze().cpu().numpy()])
    test_labels = np.array(test_labels).ravel()
    test_predictions = np.array(test_predictions).ravel()
    print(test_labels)
    print(test_predictions)
    auc = roc_auc_score(test_labels, test_predictions, average='macro', multi_class='ovr')
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, (test_predictions>=0.5).astype(int), average='macro')
    acc = accuracy_score(test_labels, (test_predictions>=0.5).astype(int))
    results = {}
    results["auc"] = auc
    results["acc"] = acc
    results["precision"] = precision
    results["recall"] = recall
    results["f1"] = f1
    results["loss"] = total_loss / len(test_df)
    return results

def main():
    parser = argparse.ArgumentParser(description='Train DSMIL on 20x patch features learned by SimCLR')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes [2]')
    parser.add_argument('--feats_size', default=1024, type=int, help='Dimension of the feature size [512]')
    parser.add_argument('--lr', default=0.0002, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--num_epochs', default=200, type=int, help='Number of total training epochs [40|200]')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--weight_decay', default=5e-3, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--dataset', default='TCGA-lung-default', type=str, help='Dataset folder name')
    parser.add_argument('--split', default=0.2, type=float, help='Training/Validation split [0.2]')
    parser.add_argument('--model', default='dsmil', type=str, help='MIL model [dsmil]')
    parser.add_argument('--dropout_patch', default=0, type=float, help='Patch dropout rate [0]')
    parser.add_argument('--dropout_node', default=0, type=float, help='Bag classifier dropout rate [0]')
    parser.add_argument('--non_linearity', action="store_true", help='Additional nonlinear operation [0]')
    parser.add_argument('--average', action="store_true", help='Average the score of max-pooling and bag aggregating')
    # unified dataset settings
    parser.add_argument('--csv_path', default='', type=str)  # csv file to describe the data split
    parser.add_argument('--feature_dir', default='', type=str)  # feature dir
    parser.add_argument('--fold', default=1, type=int) # val fold
    parser.add_argument('--nfolds', default=5, type=int) # number of folds
    # save dir
    parser.add_argument('--save_dir', default="logs") # number of folds
    args = parser.parse_args()
    gpu_ids = tuple(args.gpu_index)
    # os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(x) for x in gpu_ids)
    set_random_seed(2023, True)
    
    if args.model == 'dsmil':
        import dsmil as mil
    elif args.model == 'abmil':
        import abmil as mil
    
    i_classifier = mil.FCLayer(in_size=args.feats_size, out_size=args.num_classes).cuda()
    b_classifier = mil.BClassifier(input_size=args.feats_size, output_class=args.num_classes, dropout_v=args.dropout_node, nonlinear=args.non_linearity).cuda()
    milnet = mil.MILNet(i_classifier, b_classifier).cuda()
    if args.model == 'dsmil':
        state_dict_weights = torch.load('init.pth')
        info = milnet.load_state_dict(state_dict_weights, strict=False)
        print(info)
    criterion = nn.BCEWithLogitsLoss()
    
    optimizer = torch.optim.Adam(milnet.parameters(), lr=args.lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, 0.000005)
    
    train_set, val_set, test_set = [], [], []
    train_folds = [i for i in range(1, args.nfolds+1)]
    train_folds.remove(args.fold)
    val_folds = [args.fold]
    test_folds = [0]
    with open(args.csv_path, "r") as f:
        for row in f.readlines():
            name, label, fold, path = row.strip().split(",")
            if int(fold) in train_folds:
                train_set.append([os.path.join(args.feature_dir, name+".npz"), int(label)])
            elif int(fold) in val_folds:
                val_set.append([os.path.join(args.feature_dir, name+".npz"), int(label)])
            else:
                test_set.append([os.path.join(args.feature_dir, name+".npz"), int(label)])
    train_path = pd.DataFrame(train_set)
    val_path = pd.DataFrame(val_set)
    test_path = pd.DataFrame(test_set)

    best_score = 0
    os.makedirs(args.save_dir, exist_ok=True)

    log_file = open(os.path.join(args.save_dir, "log.txt"), 'a')
    print_log(json.dumps(vars(args).copy()), log_file)
    tb = SummaryWriter(args.save_dir)
    meters = {"loss":AverageMeter(20), "bag_loss":AverageMeter(20), "max_loss":AverageMeter(20)}

    for epoch in range(1, args.num_epochs):
        train_path = shuffle(train_path).reset_index(drop=True)
        val_path = shuffle(val_path).reset_index(drop=True)
        train_loss_bag = train(train_path, milnet, criterion, optimizer, args, meters, tb, epoch) # iterate all bags
        metrics = test(val_path, milnet, criterion, args)
        print_log('\r Epoch [%d/%d] train loss: %.4f test loss: %.4f, ACC: %.4f, AUC: %.4f' % 
                (epoch, args.num_epochs, train_loss_bag, metrics["loss"], metrics["acc"], metrics["auc"]), log_file) 
        scheduler.step()
        current_score = (metrics["acc"] + metrics["auc"])/2
        tb.add_scalar("val/loss", metrics["loss"], epoch*len(train_path))
        tb.add_scalar("val/acc", metrics["acc"], epoch*len(train_path))
        tb.add_scalar("val/auc", metrics["auc"], epoch*len(train_path))
        tb.add_scalar("val/mean", current_score, epoch*len(train_path))
        if current_score >= best_score:
            best_score = current_score
            save_name = os.path.join(args.save_dir, 'best.pth')
            torch.save(milnet.state_dict(), save_name)
            print_log('Best model saved at: ' + save_name, log_file)

    state_dict_weights = torch.load(os.path.join(args.save_dir, 'best.pth'))
    milnet.load_state_dict(state_dict_weights, strict=True)
    metrics = test(val_path, milnet, criterion, args)
    print_log("val : " + ", ".join("{}:{:.6f}".format(key, value) for key, value in metrics.items()), log_file)
    metrics = test(test_path, milnet, criterion, args)
    print_log("test : " + ", ".join("{}:{:.6f}".format(key, value) for key, value in metrics.items()), log_file)

if __name__ == '__main__':
    main()