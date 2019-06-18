import argparse
import os
import time
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import numpy as np
import models
from sklearn.preprocessing import StandardScaler
from data import get_dataset
from preprocess import get_transform
from utils.log import setup_logging, ResultsLog, save_checkpoint
from utils.meters import AverageMeter, accuracy
from utils.optim import OptimRegime
from utils.misc import torch_dtypes
from datetime import datetime
from ast import literal_eval

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ConvNet Training')

parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results',
                    help='results dir')
parser.add_argument('--save', metavar='SAVE', default='',
                    help='saved folder')
parser.add_argument('--dataset', metavar='DATASET', default='imagenet',
                    help='dataset name or folder')
parser.add_argument('--model', '-a', metavar='MODEL', default='alexnet',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: alexnet)')
parser.add_argument('--input_size', type=int, default=None,
                    help='image input size')
parser.add_argument('--model_config', default='',
                    help='additional architecture configuration')
parser.add_argument('--dtype', default='float',
                    help='type of tensor: ' +
                    ' | '.join(torch_dtypes.keys()) +
                    ' (default: half)')
parser.add_argument('--device', default='cuda',
                    help='device assignment ("cpu" or "cuda")')
parser.add_argument('--device_ids', default=[0], type=int, nargs='+',
                    help='device ids assignment (e.g 0 1 2 3')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--optimizer', default='Adam', type=str, metavar='OPT',
                    help='optimizer function used')
parser.add_argument('--lr', '--learning_rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', type=str, metavar='FILE',
                    help='evaluate model FILE on validation set')
parser.add_argument('--seed', default=123, type=int,
                    help='random seed (default: 123)')


# dataset
x_train = np.loadtxt("./input.txt", delimiter=",", dtype=np.float32).reshape(-1,4)


y_train = np.loadtxt("./output.txt", delimiter=",",dtype=np.float32).reshape(-1,1)
newlen = int(len(x_train)/4)*4
x_train = x_train[:newlen,:]
y_train = y_train[:newlen,:]
x_train = x_train.flatten().reshape(-1,16)
y_train = y_train.flatten().reshape(-1,4)

scaler = StandardScaler()
scaler.fit(x_train)
print ("scaler parameter " )
print (scaler.mean_)
print (scaler.scale_)
print ("prescale: ")
print (x_train[0])
x_train = scaler.transform(x_train)
new_size = (len(x_train)) - 20000
x_test = x_train[new_size:,:]
y_test = y_train[new_size:,:]
x_train = x_train[:new_size,:]
y_train = y_train[:new_size,:]

print (x_train.shape)
print (y_test.shape)
batch_size = 200

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self):
        pass
    def __getitem__(self, index):
        return (x_train[index], y_train[index])#.reshape(16,16))
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return y_train.shape[0]

class TestDataset(torch.utils.data.Dataset):
    def __init__(self):
        pass
    def __getitem__(self, index):
        return (x_test[index], y_test[index])#.reshape(16,16))
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return y_test.shape[0]
train_dataset = TrainDataset()
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

val_dataset = TestDataset()
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
def main():
    global args, best_prec1, dtype
    best_prec1 = 0
    args = parser.parse_args()
    dtype = torch_dtypes.get(args.dtype)
    torch.manual_seed(args.seed)
    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if args.evaluate:
        args.results_dir = '/tmp'
    if args.save is '':
        args.save = time_stamp
    save_path = os.path.join(args.results_dir, args.save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    setup_logging(os.path.join(save_path, 'log.txt'),
                  resume=args.resume is not '')
    results_path = os.path.join(save_path, 'results')
    results = ResultsLog(
        results_path, title='Training Results - %s' % args.save)

    logging.info("saving to %s", save_path)
    logging.debug("run arguments: %s", args)

    if 'cuda' in args.device and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.set_device(args.device_ids[0])
        cudnn.benchmark = True
    else:
        args.device_ids = None

    # create model
    logging.info("creating model %s", args.model)
    model = models.__dict__[args.model]
    model_config = {'input_size': args.input_size, 'dataset': args.dataset}

    if args.model_config is not '':
        model_config = dict(model_config, **literal_eval(args.model_config))

    model = model(**model_config)
    logging.info("created model with configuration: %s", model_config)

    # optionally resume from a checkpoint
    if args.evaluate:
        if not os.path.isfile(args.evaluate):
            parser.error('invalid checkpoint: {}'.format(args.evaluate))
        checkpoint = torch.load(args.evaluate)
        model.load_state_dict(checkpoint['state_dict'])
        logging.info("loaded checkpoint '%s' (epoch %s)",
                     args.evaluate, checkpoint['epoch'])
    elif args.resume:
        checkpoint_file = args.resume
        if os.path.isdir(checkpoint_file):
            results.load(os.path.join(checkpoint_file, 'results.csv'))
            checkpoint_file = os.path.join(
                checkpoint_file, 'model_best.pth.tar')
        if os.path.isfile(checkpoint_file):
            logging.info("loading checkpoint '%s'", args.resume)
            checkpoint = torch.load(checkpoint_file)
            args.start_epoch = checkpoint['epoch'] - 1
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logging.info("loaded checkpoint '%s' (epoch %s)",
                         checkpoint_file, checkpoint['epoch'])
        else:
            logging.error("no checkpoint found at '%s'", args.resume)

    num_parameters = sum([l.nelement() for l in model.parameters()])
    logging.info("number of parameters: %d", num_parameters)

    # regime = getattr(model, 'regime', [{'epoch': 0,
    #                                     'optimizer': args.optimizer,
    #                                     'lr': args.lr,
    #                                     'momentum': args.momentum,
    #                                     'weight_decay': args.weight_decay}])

    # define loss function (criterion) and optimizer
    criterion = getattr(model, 'criterion', nn.MSELoss)()
    criterion.to(args.device, dtype)
    model.to(args.device, dtype)

    if args.evaluate:
        model.eval()
        predicted = model(torch.from_numpy(x_test).to(args.device)).detach().cpu().numpy()
        print(sum(np.abs(predicted.flatten() - y_test.flatten())) / len(predicted.flatten()))
        print(sum(np.abs(predicted.flatten() - y_test.flatten())) / sum(y_test.flatten()))
        print ("avg rel err")
        print (sum(np.abs(predicted.flatten() - y_test.flatten())/np.abs(y_test.flatten())) / len(predicted.flatten()))
        print(y_test[:16])
        print ("predict")
        print(predicted[:16])
        print ("model parameters")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print (name, param.data)

        return

    #
    # train_data = get_dataset(args.dataset, 'train', transform['train'])
    # train_loader = torch.utils.data.DataLoader(
    #     train_data,
    #     batch_size=args.batch_size, shuffle=True,
    #     num_workers=args.workers, pin_memory=True)
    #

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay = 0.0001)

    #optimizer = OptimRegime(model.parameters(), regime)

    #logging.info('training regime: %s', regime)

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train_loss, train_prec1, train_prec5 = train(
            train_loader, model, criterion, epoch, optimizer)

        # evaluate on validation set
        val_loss, val_prec1, val_prec5 = validate(
            val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = val_prec1 > best_prec1
        best_prec1 = max(val_prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': args.model,
            'config': args.model_config,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'regime': 0
        }, is_best, path=save_path)
        logging.info('\n Epoch: {0}\t'
                     'Training Loss {train_loss:.4f} \t\t\t'
                     #'Training Prec@1 {train_prec1:.3f} \t'
                     #'Training Prec@5 {train_prec5:.3f} \t'
                     'Validation Loss {val_loss:.4f} \t\t\t'
                     #'Validation Prec@1 {val_prec1:.3f} \t'
                     #'Validation Prec@5 {val_prec5:.3f} \n'
                     .format(epoch + 1, train_loss=train_loss, val_loss=val_loss,
                             #train_prec1=train_prec1, val_prec1=val_prec1,
                             #train_prec5=train_prec5, val_prec5=val_prec5
                             ))

        results.add(epoch=epoch + 1, train_loss=train_loss, val_loss=val_loss,
                    train_error1=100 - train_prec1, val_error1=100 - val_prec1,
                    train_error5=100 - train_prec5, val_error5=100 - val_prec5)
        results.plot(x='epoch', y=['train_loss', 'val_loss'],
                     legend=['training', 'validation'],
                     title='Loss', ylabel='loss')
        results.plot(x='epoch', y=['train_error1', 'val_error1'],
                     legend=['training', 'validation'],
                     title='Error@1', ylabel='error %')
        results.plot(x='epoch', y=['train_error5', 'val_error5'],
                     legend=['training', 'validation'],
                     title='Error@5', ylabel='error %')
        results.save()


def forward(data_loader, model, criterion, epoch=0, training=True, optimizer=None):
    regularizer = getattr(model, 'regularization', None)
    if args.device_ids and len(args.device_ids) > 1:
        model = torch.nn.DataParallel(model, args.device_ids)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for i, (inputs, target) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.to(args.device)
        inputs = inputs.to(args.device, dtype=dtype)

        # compute output
        output = model(inputs)
        loss = criterion(output, target)
        if regularizer is not None:
            loss += regularizer(model)

        if type(output) is list:
            output = output[0]

        # measure accuracy and record loss
        #print (output)
        #prec1, prec5 = accuracy(output.detach(), target, topk=(1, 5))
        losses.update(float(loss), inputs.size(0))
        #top1.update(float(prec1), inputs.size(0))
        #top5.update(float(prec5), inputs.size(0))

        if training:
            #optimizer.update(epoch, epoch * len(data_loader) + i)
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        #
        # if i % args.print_freq == 0:
        #     logging.info('{phase} - Epoch: [{0}][{1}/{2}]\t'
        #                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
        #                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
        #                      epoch, i, len(data_loader),
        #                      phase='TRAINING' if training else 'EVALUATING',
        #                      batch_time=batch_time,
        #                      data_time=data_time, loss=losses, top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


def train(data_loader, model, criterion, epoch, optimizer):
    # switch to train mode
    model.train()
    return forward(data_loader, model, criterion, epoch,
                   training=True, optimizer=optimizer)


def validate(data_loader, model, criterion, epoch):
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        return forward(data_loader, model, criterion, epoch,
                       training=False, optimizer=None)


if __name__ == '__main__':
    main()