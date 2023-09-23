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

def print_log(tstr, f):
    f.write('\n')
    f.write(tstr)
    f.flush()
    print(tstr)

def get_bag_feats(csv_file_df, args): # modify for unified dataset
    feature = np.load(csv_file_df.iloc[0])["features"]
    label = csv_file_df.iloc[1]
    return np.array([label]), feature

def train(train_df, milnet, criterion, optimizer, args):
    milnet.train()
    csvs = shuffle(train_df).reset_index(drop=True)
    total_loss = 0
    bc = 0
    Tensor = torch.cuda.FloatTensor
    for i in range(len(train_df)):
        optimizer.zero_grad()
        label, feats = get_bag_feats(train_df.iloc[i], args)
        feats = dropout_patches(feats, args.dropout_patch)
        bag_label = torch.from_numpy(label).cuda().float()
        bag_feats = torch.from_numpy(feats).cuda().float()
        bag_feats = bag_feats.view(-1, args.feats_size)
        ins_prediction, bag_prediction, _, _ = milnet(bag_feats)
        max_prediction, _ = torch.max(ins_prediction, 0)        
        bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
        max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
        loss = 0.5*bag_loss + 0.5*max_loss
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item()
        sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f' % (i, len(train_df), loss.item()))
    return total_loss / len(train_df)

def dropout_patches(feats, p):
    idx = np.random.choice(np.arange(feats.shape[0]), int(feats.shape[0]*(1-p)), replace=False)
    sampled_feats = np.take(feats, idx, axis=0)
    pad_idx = np.random.choice(np.arange(sampled_feats.shape[0]), int(feats.shape[0]*p), replace=False)
    pad_feats = np.take(sampled_feats, pad_idx, axis=0)
    sampled_feats = np.concatenate((sampled_feats, pad_feats), axis=0)
    return sampled_feats

def test(test_df, milnet, criterion, args): # modified for unified eval metrics
    milnet.eval()
    csvs = shuffle(test_df).reset_index(drop=True)
    total_loss = 0
    test_labels = []
    test_predictions = []
    Tensor = torch.cuda.FloatTensor
    with torch.no_grad():
        for i in range(len(test_df)):
            label, feats = get_bag_feats(test_df.iloc[i], args)
            bag_label = torch.from_numpy(label).cuda().float()
            bag_feats = torch.from_numpy(feats).cuda().float()
            bag_feats = bag_feats.view(-1, args.feats_size)
            ins_prediction, bag_prediction, _, _ = milnet(bag_feats)
            max_prediction, _ = torch.max(ins_prediction, 0)  
            bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
            max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
            loss = 0.5*bag_loss + 0.5*max_loss
            total_loss = total_loss + loss.item()
            sys.stdout.write('\r Testing bag [%d/%d] bag loss: %.4f' % (i, len(test_df), loss.item()))
            test_labels.extend([label])
            if args.average:
                test_predictions.extend([(0.5*torch.sigmoid(max_prediction)+0.5*torch.sigmoid(bag_prediction)).squeeze().cpu().numpy()])
            else: test_predictions.extend([(0.0*torch.sigmoid(max_prediction)+1.0*torch.sigmoid(bag_prediction)).squeeze().cpu().numpy()])
    test_labels = np.array(test_labels).ravel()
    test_predictions = np.array(test_predictions).ravel()
    auc = roc_auc_score(test_labels, test_predictions, average='macro', multi_class='ovr')
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, test_predictions>=0.5, average='macro')
    acc = accuracy_score(test_labels, test_predictions>=0.5)
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
    parser.add_argument('--non_linearity', default=1, type=float, help='Additional nonlinear operation [0]')
    parser.add_argument('--average', type=bool, default=True, help='Average the score of max-pooling and bag aggregating')
    # unified dataset settings
    parser.add_argument('--csv_path', default='', type=str)  # csv file to describe the data split
    parser.add_argument('--feature_dir', default='', type=str)  # feature dir
    parser.add_argument('--fold', default=1, type=int) # val fold
    parser.add_argument('--nfolds', default=5, type=int) # number of folds
    # save dir
    parser.add_argument('--save_dir', default="logs") # number of folds
    args = parser.parse_args()
    gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(x) for x in gpu_ids)
    
    if args.model == 'dsmil':
        import dsmil as mil
    elif args.model == 'abmil':
        import abmil as mil
    
    i_classifier = mil.FCLayer(in_size=args.feats_size, out_size=args.num_classes).cuda()
    b_classifier = mil.BClassifier(input_size=args.feats_size, output_class=args.num_classes, dropout_v=args.dropout_node, nonlinear=args.non_linearity).cuda()
    milnet = mil.MILNet(i_classifier, b_classifier).cuda()
    if args.model == 'dsmil':
        state_dict_weights = torch.load('init.pth')
        try:
            milnet.load_state_dict(state_dict_weights, strict=False)
        except:
            del state_dict_weights['b_classifier.v.1.weight']
            del state_dict_weights['b_classifier.v.1.bias']
            milnet.load_state_dict(state_dict_weights, strict=False)
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

    for epoch in range(1, args.num_epochs):
        train_path = shuffle(train_path).reset_index(drop=True)
        val_path = shuffle(val_path).reset_index(drop=True)
        train_loss_bag = train(train_path, milnet, criterion, optimizer, args) # iterate all bags
        metrics = test(val_path, milnet, criterion, args)
        print_log('\r Epoch [%d/%d] train loss: %.4f test loss: %.4f, ACC: %.4f, AUC: %.4f' % 
                (epoch, args.num_epochs, train_loss_bag, metrics["loss"], metrics["acc"], metrics["auc"]), log_file) 
        scheduler.step()
        current_score = (metrics["acc"] + metrics["auc"])/2
        if current_score >= best_score:
            best_score = current_score
            save_name = os.path.join(args.save_dir, 'best.pth')
            torch.save(milnet.state_dict(), save_name)
            print_log('Best model saved at: ' + save_name, log_file)

    metrics = test(val_path, milnet, criterion, args)
    print_log("val : " + ", ".join("{}:{:.6f}".format(key, value) for key, value in metrics.items()), log_file)
    metrics = test(test_path, milnet, criterion, args)
    print_log("test : " + ", ".join("{}:{:.6f}".format(key, value) for key, value in metrics.items()), log_file)

if __name__ == '__main__':
    main()