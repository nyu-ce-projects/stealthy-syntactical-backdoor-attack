
import argparse

from Trainer.BaseTrainer import BaseTrainer
from Models.BERT import BERT
from Models.LSTM import LSTM
from Dataset import OLID,SST2,AG


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Syntactical Poisoning')
    parser.add_argument('--data', type=str, default='sst-2')
    parser.add_argument('--model','-m', type=str, default='BERT-IT')
    parser.add_argument('--lr', default=2e-5, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true',help='resume from checkpoint')
    parser.add_argument('--cpu', '-c', action='store_true',help='Use CPU only')
    parser.add_argument('--workers', '-w',default=2, type=int,help='no of workers')    
    parser.add_argument('--epochs', '-e',default=10, type=int,help='Epochs')
    parser.add_argument('--warmup_epochs', type=int, default=3)
    parser.add_argument('--optim', '-o',default='sgd', type=str,help='optimizer type')
    parser.add_argument('--batchsize', '-bs',default=32, type=int,help='Batch Size')    
    parser.add_argument('--transfer', type=bool, default=False)
    parser.add_argument('--transfer_epoch', type=int, default=3)

    args = parser.parse_args()

    if args.data=='sst-2':
        dataset = SST2.SST2
    elif args.data=='ag':
        dataset = AG.AG
    elif args.data=='olid':
        dataset = OLID.OLID

    if args.model=='BERT-IT' and args.data=='sst-2':
        model = BERT()
        dataset = SST2.SST2Bert
    elif args.model=='BERT-IT' and args.data=='ag':
        model = BERT(True)
        dataset = AG.AGBert
    elif args.model=='BERT-IT' and args.data=='olid':
        model = BERT()
        dataset = OLID.OLIDBert
    elif args.model=='LSTM' and args.data=='sst-2':
        model = LSTM()
        dataset = SST2.SST2
    elif args.model=='LSTM' and args.data=='ag':
        model = LSTM(True)
        dataset = AG.AG
    elif args.model=='LSTM' and args.data=='olid':
        model = LSTM()
        dataset = OLID.OLID
    # elif args.model=='BERT-CFT':
    #     model = BERT()

    net = BaseTrainer(dataset,model,args)
    net.train()
    
    print("Backdoor Training Completed")
