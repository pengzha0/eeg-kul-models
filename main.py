import os
import statistics
import mne
import argparse
import numpy as np
from torch.utils.data import TensorDataset, DataLoader,random_split
import torch.nn as nn
from sklearn.metrics import accuracy_score
import xgboost as xgb

# from keras.callbacks import EarlyStopping
from eutils.kflod import five_fold
import scipy
from models import *
from utils import *
import random
from preproc.util import list_data_split, normalize_data
from sklearn.model_selection import KFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB



def set_seed(seed):
    # 设置 Python 内置的随机数生成器的种子
    random.seed(seed)
    
    # 设置 NumPy 随机数生成器的种子
    np.random.seed(seed)
    
    # 设置 PyTorch 随机数生成器的种子
    torch.manual_seed(seed)
    
    # 如果使用 GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果有多个 GPU
        
    # 确保在 CUDA 上进行操作的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 调用函数设置随机种子
set_seed(520520)

time_len = 1
sampling_rate = 128
sample_len, channels_num = int(sampling_rate * time_len), 64
overlap_rate = 0.5
window_sliding = int(sample_len * time_len * overlap_rate)
freq_bands = [[1, 50]]

parser = argparse.ArgumentParser(description='eeg competition for CS')


parser.add_argument('--datadir', type=str, default='/root/autodl-tmp/kul', help='dir to the dataset or the validation set')
parser.add_argument('--opt', type=str, default='adam', help='optimizer [adam, sgd]')
parser.add_argument('--model', type=str, default='cacnn',choices=['cacnn','cnn_lstm','gcn','resnet18','eegnet','eegnet_','eegnet_bnn','inception','deepnetwork','pyramidnet','XGBoost','AdaBoost','Decision_Tree','Random_Forest','Gaussian_Naive_Bayes'], help='model to train the dataset')
parser.add_argument('--subject_id',type =int,default=0)
parser.add_argument('--batch_size', type=int, default=128, help='train and val batch')
parser.add_argument('--epochs', type=int, default=60, help='epochs')
parser.add_argument('--savedir', type=str, default='./results', help='dir to the results')
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for all weights')
parser.add_argument('--lr-type', type=str, default='cosine', help='learning rate strategy [cosine, multistep]')
parser.add_argument('--wd', type=float, default=1e-5, help='weight decay for all weights')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum (default: 0.9)')
parser.add_argument('--print-freq', type=int, default=10, help='print frequency (default: 10)')
parser.add_argument('--save-freq', type=int, default=30, help='save frequency (default: 10)')
parser.add_argument('--label-smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--warm-epoch', default=0, type=int, help='epoch number to warm up')
parser.add_argument('--evaluate', type=str, default=None, help="full path to checkpoint to be evaluated or 'best'")
parser.add_argument('--lr-steps', type=str, default="60-85", help='steps for multistep learning rate')
parser.add_argument('--lr-gammas', type=str, default=None, help='corresponding gammas for lr_steps to reduce lr')
args = parser.parse_args()


def preprocess_data_ten(sub_id,  data_folder="DATA"):
    """
    Preprocesses EEG data for a given subject, filtering and resampling the data according to specified frequency bands.
    """

    data_path = os.path.join(data_folder,f'S{sub_id}.mat')
    data_mat = scipy.io.loadmat(data_path)
    eeg_d = []
    label = []
    for k_tra in range(20):
        tmp_eeg = data_mat['trials'][0, k_tra]['RawData'][0, 0]['EegData'][0, 0]
        lab = 0 if str(data_mat['trials'][0, k_tra]['attended_ear'][0, 0][0]) == 'L' else 1

        # times = np.arange(tmp_eeg.shape[0]) / 128.  # 假设原始采样率为 128Hz
        # 创建一个 MNE Raw 对象
        info = mne.create_info(64, 128., 'eeg')  # 64 通道，128Hz 采样率
        raw = mne.io.RawArray(tmp_eeg.T, info)  # 注意，data 需要转置，因为 MNE 要求通道在第一维

        # 进行带通滤波
        raw.filter(l_freq=0.1, h_freq=63.9)

        # 进行陷波滤波
        raw.notch_filter(50.0)

        # 进行重采样
        raw.resample(200, n_jobs=5)

        # 将处理后的数据转换回 numpy 数组
        data_resampled = np.transpose(raw.get_data())
        eeg_d.append(data_resampled)
        label.append(lab)

    # Return the preprocessed data and labels
    return eeg_d, label

def load_data(subject_id):
    all_data=[]
    all_label=[]
    for id in subject_id:
    # Initialize dictionaries to store data, labels, and fold indices for all subjects


    # Iterate over each subject to preprocess and partition their data

        sub_id = f'{id + 1}'

    # Load and preprocess data for the subject
    # Data is expected to be in the form of trail * time * channels, with a sampling frequency of 128Hz
        data, label = preprocess_data_ten(data_folder=args.datadir, sub_id=sub_id)

        # Partition the data into training and validation sets
        data, label, split_index = list_data_split(
            data, None, label, time_len, window_sliding, 
            sampling_rate=sampling_rate
        )

    # Normalize the data to have zero mean and unit variance
        data = [normalize_data(d) for d in data]
        all_data.append(np.concatenate(data, axis=0))
        all_label.append(np.concatenate(label, axis=0))

    all_subjects_data= torch.from_numpy(np.concatenate(all_data, axis=0)).float()
    all_subjects_label = torch.from_numpy(np.concatenate(all_label, axis=0)).float()

    return all_subjects_data,all_subjects_label

def within_load_data(subject_id):

    all_subjects_data,all_subjects_label = load_data(subject_id)
    train_percent = 0.8
    train_size = int(train_percent * len(all_subjects_data))

    # 随机划分训练集和测试集
    train_dataset, test_dataset = random_split(TensorDataset(all_subjects_data, all_subjects_label), [train_size, len(all_subjects_data) - train_size])
    # 创建Tensor数据集
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

    val_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader,val_loader


def without_trial_load_data(subject_id):

    all_subjects_data, all_subjects_label = load_data(subject_id)
    train_percent = 0.8
    train_size = int(train_percent * len(all_subjects_data))

    # 划分训练集和测试集
    train_data = all_subjects_data[:train_size]
    train_label = all_subjects_label[:train_size]

    test_data = all_subjects_data[train_size:]
    test_label = all_subjects_label[train_size:]

    # 创建Tensor数据集
    train_dataset = TensorDataset(train_data, train_label)
    test_dataset = TensorDataset(test_data, test_label)

    # 创建数据加载器
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader,val_loader


def without_subject_load_data(train_id,test_id):

    train_data,train_label = load_data(train_id)

    test_data,test_label = load_data(test_id)

    # 随机划分训练集和测试集
    train_dataset = TensorDataset(train_data,train_label)
    test_dataset = TensorDataset(test_data,test_label)

    # 创建Tensor数据集
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

    val_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader,val_loader


def main(running_file):
    all_data, all_labels = load_data([args.subject_id])
    criterion = nn.CrossEntropyLoss()
    criterion_smooth = CrossEntropyLabelSmooth(2, args.label_smooth)

    best_prec1 = 0.0
    saveID = None
    kf = KFold(n_splits=5, shuffle=True, random_state=520520) 
    fold_idx = 1
    all_prec1_scores = []
    
    for train_idx, val_idx in kf.split(all_data):

        # Create train and validation loaders for this fold
        train_data = all_data[train_idx]
        train_label = all_labels[train_idx]
        val_data = all_data[val_idx]
        val_label = all_labels[val_idx]

        train_dataset = TensorDataset(train_data, train_label)
        val_dataset = TensorDataset(val_data, val_label)

        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False)

        best_prec1 = 0.0


        if args.model == "cacnn":
            model = CACNN(channels_num=64, sample_len=128, is_attention=True)
        elif args.model == "eegnet":
            model = EEGNet(channels_num=64)
        elif args.model == "eegnet_":
            model = EEGNet_(channels_num=64)
        elif args.model == "eegnet_bnn":
            model = EEGNet_bnn(channels_num=64)
        elif args.model == "resnet18":
            model = ResNet18(channels_num=64,sample_len=128)
        elif args.model == "inception":
            model = INCEPTION(channels_num=64,sample_len=128)
        elif args.model == "pyramidnet":
            model = PYRAMIDNET(channels_num=64,sample_len=128)
        elif args.model == "gcn":
            # Set the model parameters and create the model
            graph_layer_num = 3
            graph_convolution_kernel = 16
            is_channel_attention = False
            model = GCN(channels_num, sample_len, graph_layer_num, graph_convolution_kernel, is_channel_attention)
        elif args.model == "cnn_lstm":
            model = CNN_LSTM(channels_num=64)     
        
        
        if args.model == "XGBoost":
            train_data_reshaped = train_data.numpy().reshape(train_data.shape[0], -1)
            val_data_reshaped = val_data.numpy().reshape(val_data.shape[0], -1)
            train_label_numpy = train_label.numpy()
            val_label_numpy = val_label.numpy()
            # 创建DMatrix对象
            dtrain = xgb.DMatrix(train_data_reshaped, label=train_label_numpy)
            dval = xgb.DMatrix(val_data_reshaped, label=val_label_numpy)

            # 设置参数：二分类问题
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 200,
                'use_label_encoder': False
            }

            # 训练模型
            model = xgb.train(params, dtrain, num_boost_round=100, evals=[(dval, 'validation')])

        elif args.model == "AdaBoost":
            # 将 train_data 和 val_data 转换为 numpy 并展平
            train_data_reshaped = train_data.numpy().reshape(train_data.shape[0], -1)
            val_data_reshaped = val_data.numpy().reshape(val_data.shape[0], -1)

            # 将标签也转换为 numpy 格式
            train_label_numpy = train_label.numpy()
            val_label_numpy = val_label.numpy()
            # 初始化一个浅层决策树作为基学习器
            estimator  = DecisionTreeClassifier(max_depth=1)

            # 初始化 AdaBoost 模型
            ada_model = AdaBoostClassifier(estimator =estimator , n_estimators=50, learning_rate=0.1)

            # 训练模型
            ada_model.fit(train_data_reshaped, train_label_numpy)


    
        elif args.model == "Decision_Tree":
            train_data_reshaped = train_data.numpy().reshape(train_data.shape[0], -1)
            val_data_reshaped = val_data.numpy().reshape(val_data.shape[0], -1)

            # 将标签也转换为 numpy 格式
            train_label_numpy = train_label.numpy()
            val_label_numpy = val_label.numpy()
            # 初始化决策树模型
            dt_model = DecisionTreeClassifier(max_depth=6)  # 你可以调整 max_depth 来控制树的复杂度

            # 训练模型
            dt_model.fit(train_data_reshaped, train_label_numpy)
        elif args.model == "Random_Forest":
            train_data_reshaped = train_data.numpy().reshape(train_data.shape[0], -1)
            val_data_reshaped = val_data.numpy().reshape(val_data.shape[0], -1)

            # 将标签也转换为 numpy 格式
            train_label_numpy = train_label.numpy()
            val_label_numpy = val_label.numpy()
            # 初始化随机森林模型
            rf_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)

            # 训练模型
            rf_model.fit(train_data_reshaped, train_label_numpy)
        elif args.model == "Gaussian_Naive_Bayes":
            train_data_reshaped = train_data.numpy().reshape(train_data.shape[0], -1)
            val_data_reshaped = val_data.numpy().reshape(val_data.shape[0], -1)

            # 将标签也转换为 numpy 格式
            train_label_numpy = train_label.numpy()
            val_label_numpy = val_label.numpy()
            # 初始化 Gaussian Naive Bayes 模型
            gnb_model = GaussianNB()

            # 训练模型
            gnb_model.fit(train_data_reshaped, train_label_numpy)

        print(f"Fold {fold_idx}...")
        if args.model == "XGBoost":
            # 预测并评估
            val_preds = model.predict(dval)
            val_preds_binary = [1 if pred > 0.5 else 0 for pred in val_preds]

            # 计算准确率
            val_prec1 = accuracy_score(val_label_numpy, val_preds_binary)
            best_prec1 = max(val_prec1, best_prec1)
            print(f'验证集准确率: {val_prec1 * 100:.2f}%')    

        elif args.model == "AdaBoost":
            # 在验证集上进行预测
            val_preds = ada_model.predict(val_data_reshaped)

            # 计算准确率
            val_prec1 = accuracy_score(val_label_numpy, val_preds)
            best_prec1 = max(val_prec1, best_prec1)
            print(f'验证集准确率: {val_prec1 * 100:.2f}%')

        elif args.model == "Decision_Tree":
            # 在验证集上进行预测
            val_preds = dt_model.predict(val_data_reshaped)

            # 计算准确率
            val_prec1 = accuracy_score(val_label_numpy, val_preds)
            best_prec1 = max(val_prec1, best_prec1)
            print(f'验证集准确率: {val_prec1 * 100:.2f}%')

        elif args.model == "Random_Forest":
                        # 在验证集上进行预测
            val_preds = rf_model.predict(val_data_reshaped)

            val_prec1 = accuracy_score(val_label_numpy, val_preds)
            best_prec1 = max(val_prec1, best_prec1)
            print(f'验证集准确率: {val_prec1 * 100:.2f}%')

        elif args.model == "Gaussian_Naive_Bayes":
            val_preds = gnb_model.predict(val_data_reshaped)

            val_prec1 = accuracy_score(val_label_numpy, val_preds)
            best_prec1 = max(val_prec1, best_prec1)
            print(f'验证集准确率: {val_prec1 * 100:.2f}%')
        else:
            model = model.cuda()
            if args.opt == 'adam':
                optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.wd)
            elif args.opt == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(),lr=args.lr,weight_decay=args.wd, momentum=args.momentum)
            for epoch in range(args.epochs):
                lr_str = adjust_learning_rate(optimizer, epoch, args, method=args.lr_type)
                train_prec1, loss = train(train_loader, model, criterion, optimizer, epoch, running_file,lr_str, args)
                val_prec1 = validate(val_loader, model, criterion, args)

                is_best = val_prec1 >= best_prec1
                best_prec1 = max(val_prec1, best_prec1)

                if is_best:
                    print(f"New best precision at fold {fold_idx}, epoch {epoch}: {best_prec1}")

        all_prec1_scores.append(best_prec1)
        fold_idx += 1

    avg_prec1 = np.mean(all_prec1_scores)
    print(f"Average Prec@1 across all folds: {avg_prec1}")

    # for epoch in range( args.epochs):
        
    #     lr_str = adjust_learning_rate(optimizer, epoch, args, method=args.lr_type)
    #     tr_prec1, loss = \
    #         train(train_loader, model, criterion, optimizer, epoch, 
    #                 running_file, lr_str, args)
    #     val_prec1 = validate(val_loader, model, criterion, args)
        
    #     is_best = val_prec1 >= best_prec1
    #     best_prec1 = max(val_prec1, best_prec1)

    #     log = ("Epoch %03d/%03d: top1 %.4f " + \
    #         " | train-top1 %.4f |  loss %.4f | lr %s | Time %s\n") \
    #         % (epoch, args.epochs, val_prec1, tr_prec1, \
    #         loss, lr_str, time.strftime('%Y-%m-%d %H:%M:%S'))
    #     with open(log_file, 'a') as f:
    #         f.write(log)


    #     print('checkpoint saving in local rank 0')
    #     running_file.write('checkpoint saving in local rank 0\n')
    #     running_file.flush()
        
    #     saveID = save_checkpoint({
    #         'epoch': epoch,
    #         'state_dict': model.state_dict(),
    #         'best_prec1': best_prec1,
    #         'optimizer': optimizer.state_dict(),
    #         }, epoch, args.savedir, is_best, 
    #         saveID, keep_freq=args.save_freq)

    with open(f'/home/kul/results/{args.model}.csv','a') as f:
        f.write(f"{args.subject_id},{avg_prec1}\n")


def train(train_loader, model, criterion, optimizer, epoch, 
        running_file, running_lr, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter('sum')
    top5 = AverageMeter('sum')

    ## Switch to train mode
    model.train()

    running_file.write('\n%s\n' % str(args))
    running_file.flush()

    wD = len(str(len(train_loader)))
    wE = len(str(args.epochs))

    end = time.time()

    # change the parameter of recu and fda

    for i, (input, target) in enumerate(train_loader):

        ## Measure data loading time
        data_time.update(time.time() - end)


        input = input.cuda()
        target = target.cuda().long()

        ## Compute output
        output = model(input)
        loss = criterion(output, target)

        ## Measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        ## Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # clamp real_weights and bconv weights, not other weights (bnn, relu, sign...)
        #clip(optimizer, args.clip)

        ## Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        ## Record
        if i % args.print_freq == 0:
            runinfo = str((' Epoch: [{0:0%dd}/{1:0%dd}][{2:0%dd}/{3:0%dd}]\t' \
                      % ( wE, wE, wD, wD) + \
                      'Time {batch_time.val:.3f}\t' + \
                      'Data {data_time.val:.3f}\t' + \
                      'Loss {loss.val:.4f}\t' + \
                      'Prec@1 {top1.val100:.3f}\t' + \
                      'lr {lr}\t').format(
                          epoch, args.epochs, i, len(train_loader), 
                          batch_time=batch_time, data_time=data_time, 
                          loss=losses, top1=top1,lr=running_lr))
            print(runinfo)

            running_file.write('%s\n' % runinfo)
            running_file.flush()

    return top1.avg100, losses.avg


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter('sum')
    top5 = AverageMeter('sum')
    import copy
    ## Switch to evaluate mode
    # test_model = copy.deepcopy(model)
    model.eval()
    # model = test_model
    # compare_models(model,test_model)
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        with torch.no_grad():
            target = target.cuda().long()
            input = input.cuda()
            
            ## Compute output
            output = model(input)
            loss = criterion(output, target)

        ## Measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        ## Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        ## Record
        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val100:.3f} ({top1.avg100:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))

    print(' * Prec@1 {top1.avg100:.3f}'.format(top1=top1))

    return top1.avg100


if __name__ == '__main__':
    """
    The main entry point of the script that processes the results from the main function,
    calculates the accuracy & standard deviation, and saves the results & average results.
    """

    args.savedir = f"{args.savedir}/{args.model}/{args.subject_id}_{time.strftime('%Y-%m-%d-%H-%M-%S')}"
    os.makedirs(args.savedir, exist_ok=True)

    log_file = os.path.join(args.savedir, '%s_log.txt' % args.model)
    running_file = os.path.join(args.savedir, '%s_running-%s.txt' % (args.model, time.strftime('%Y-%m-%d-%H-%M-%S')))

    with open(running_file, 'w') as f:
        main(f)
