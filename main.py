
import argparse

from Trainer.BaseTrainer import BaseTrainer
# from Trainer.BertTrainer import BertTrainer
from Trainer.LSTMTrainer import LSTMTrainer
from Models.BERT import BERT
from Models.LSTM import LSTM
from Dataset import OLID,SST2,AG
from utils import get_vocab,read_data
from config import SST2DataPath,AGDataPath,OLIDDataPath
from GenPoisonData import PoisonDataGenerator



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Syntactical Poisoning')
    parser.add_argument('--data', type=str, default='sst-2')
    parser.add_argument('--model','-m', type=str, default='BERT')
    parser.add_argument('--cft', type=bool, default=False) # for bert, using clean fine tuning at later stage
    parser.add_argument('--cft_epochs', type=int, default=3)
    parser.add_argument('--lr', default=2e-5, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true',help='resume from checkpoint')
    parser.add_argument('--cpu', '-c', action='store_true',help='Use CPU only')
    parser.add_argument('--workers', '-w',default=2, type=int,help='no of workers')    
    parser.add_argument('--epochs', '-e',default=10, type=int,help='Epochs')
    parser.add_argument('--warmup_epochs', type=int, default=3)
    parser.add_argument('--optim', '-o',default='AdamW', type=str,help='optimizer type')
    parser.add_argument('--batchsize', '-bs',default=32, type=int,help='Batch Size')    
    parser.add_argument('--transfer', type=bool, default=False)
    parser.add_argument('--transfer_epoch', type=int, default=3)
    parser.add_argument('--poison_gen', type=bool, default=False)

    args = parser.parse_args()
    
    if args.poison_gen:
        print("Poison Data Generation Starts")
        PoisonDataObj = PoisonDataGenerator(args)
        PoisonDataObj.generator()
        print("Poison Data Generation Ends")

    if args.model=='BERT' and args.data=='sst-2':
        model = BERT()
        dataset = SST2.SST2Bert
        trainer = BaseTrainer(dataset,model,args)
    elif args.model=='BERT' and args.data=='ag':
        model = BERT(num_labels=4)
        dataset = AG.AGBert
        trainer = BaseTrainer(dataset,model,args)
    elif args.model=='BERT' and args.data=='olid':
        model = BERT()
        dataset = OLID.OLIDBert
        trainer = BaseTrainer(dataset,model,args)
    elif args.model=='LSTM' and args.data=='sst-2':
        vocab_size = len(get_vocab(read_data(SST2DataPath,'train',True)))
        model = LSTM(vocab_size=vocab_size)
        dataset = SST2.SST2
        trainer = LSTMTrainer(dataset,model,args)
    elif args.model=='LSTM' and args.data=='ag':
        vocab_size = len(get_vocab(read_data(AGDataPath,'train',True)))
        model = LSTM(vocab_size=vocab_size,num_labels=4)
        dataset = AG.AG
        trainer = LSTMTrainer(dataset,model,args)
    elif args.model=='LSTM' and args.data=='olid':
        vocab_size = len(get_vocab(read_data(OLIDDataPath,'train',True)))
        model = LSTM(vocab_size=vocab_size)
        dataset = OLID.OLID
        trainer = LSTMTrainer(dataset,model,args)

    
    trainer.train()
    
    print("Backdoor Training Completed")

    # if args.model=='BERT' and args.data=='sst-2':
    #     trainer = BertTrainer(SST2.SST2Bert,BERT,args)
    # elif args.model=='BERT' and args.data=='ag':
    #     # model = BERT(True) # TODO
    #     trainer = BertTrainer(AG.AGBert,BERT,args)
    # elif args.model=='BERT' and args.data=='olid':
    #     trainer = BertTrainer(OLID.OLIDBert,BERT,args)
    # elif args.model=='LSTM' and args.data=='sst-2':
    #     trainer = LSTMTrainer(SST2.SST2,LSTM,args)
    # elif args.model=='LSTM' and args.data=='ag':
    #     trainer = LSTMTrainer(AG.AG,LSTM,args)
    # elif args.model=='LSTM' and args.data=='olid':
    #     trainer = LSTMTrainer(OLID.OLID,LSTM,args)