
'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import transformers

import os

# from Models.BERT import BERT
# from Dataset.OLID import OLID

import time

class BaseTrainer():
    def __init__(self,Dataset,Model,args) -> None:
        print(args)
        self.args = args
        self.lr = self.args.lr
        self.optim = self.args.optim
        self.num_workers = args.workers
        self.epochs = self.args.epochs
        self.batch_size = self.args.batchsize
        self.warmup_epochs = self.args.warmup_epochs
        self.n_gpus = 1
        self.dataset = Dataset
        self.set_device()
        self.build_model(Model)
        self.totalTrainableParams = 0
        self.trainableParameters = self.net.parameters()
        self.totalTrainableParams += sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        
        self.load_dataset()
        self.setup_optimizer_losses()
        
        self.best_acc = 0

    def set_device(self):
        if self.args.cpu is not False:
            self.device = 'cpu'
        else:
            if torch.cuda.is_available():
                self.device = 'cuda' 
                self.n_gpus = torch.cuda.device_count()
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
            
        print(self.device)

    def load_dataset(self):
        
        # Data
        print('==> Preparing data..')
        
        clean_train_dataset = self.dataset('train', False)
        self.clean_train_loader = torch.utils.data.DataLoader(clean_train_dataset, batch_size=self.batch_size*self.n_gpus, shuffle=True, num_workers=self.num_workers)

        clean_test_dataset = self.dataset('test', False)
        self.clean_test_loader = torch.utils.data.DataLoader(clean_test_dataset, batch_size=self.batch_size*self.n_gpus, shuffle=False, num_workers=self.num_workers)

        clean_dev_dataset = self.dataset('dev', False)
        self.clean_dev_loader = torch.utils.data.DataLoader(clean_dev_dataset, batch_size=self.batch_size*self.n_gpus, shuffle=False, num_workers=self.num_workers)

        poison_train_dataset = self.dataset('train', True)
        self.poison_train_loader = torch.utils.data.DataLoader(poison_train_dataset, batch_size=self.batch_size*self.n_gpus, shuffle=True, num_workers=self.num_workers)
        
        poison_test_dataset = self.dataset('test', True)
        self.poison_test_loader = torch.utils.data.DataLoader(poison_test_dataset, batch_size=self.batch_size*self.n_gpus, shuffle=False, num_workers=self.num_workers)

        poison_dev_dataset = self.dataset('dev', True)
        self.poison_dev_loader = torch.utils.data.DataLoader(poison_dev_dataset, batch_size=self.batch_size*self.n_gpus, shuffle=False, num_workers=self.num_workers)
        
    def setup_optimizer_losses(self):
        self.criterion = nn.CrossEntropyLoss()
        if self.optim=='SGD':
            self.optimizer = optim.SGD(self.trainableParameters, lr=self.lr,momentum=0.9, weight_decay=5e-4)
        elif self.optim=='SGDN':
            self.optimizer = optim.SGD(self.trainableParameters, lr=self.lr,momentum=0.9, weight_decay=5e-4,nesterov=True)
        else:
            self.optimizer = eval("optim."+self.optim)(self.trainableParameters, lr=self.lr, weight_decay=5e-4)
        print(self.optimizer) 
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)
        num_warmup_steps = self.warmup_epochs * len(self.poison_train_loader)
        num_training_steps = (self.warmup_epochs+self.epochs) * len(self.poison_train_loader)
        self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer,num_warmup_steps=num_warmup_steps,num_training_steps=num_training_steps)  # TODO: try get_polynomial_decay_schedule_with_warmup,cosine

    def build_model(self,Model):
        # Model
        print('==> Building model..')
        self.net = Model()
        self.net = self.net.to(self.device)
        if self.device in ['cuda','mps']:
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True

    def train(self):
        try:
            print("Total Trainable Parameters : {}".format(self.totalTrainableParams))
            # print("Total Steps : {}".format(len(self.trainloader)))
            # startTime = time.perf_counter()
            model_version_name = int(time.time())
            for epoch in range(self.epochs+self.warmup_epochs):
                self.train_epoch(epoch)
                poison_success_rate_dev,_ = self.evaluate(epoch,self.poison_dev_loader)
                clean_acc,_ = self.evaluate(epoch,self.clean_dev_loader)
                print('attack success rate in dev: {}; clean acc in dev: {}'.format(poison_success_rate_dev, clean_acc))
                # self.saveCheckpoint(epoch,acc)
                self.scheduler.step()
                self.saveModel(model_version_name,epoch)
                print('*' * 89)
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')
        
        poison_success_rate_test,_ = self.evaluate(epoch,self.poison_test_loader)
        clean_acc,_ = self.evaluate(epoch,self.clean_test_loader)
        print('*' * 89)
        print('finish all, attack success rate in test: {}, clean acc in test: {}'.format(poison_success_rate_test, clean_acc))

    def train_epoch(self,epoch):
        self.net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (padded_text, attention_masks, labels) in enumerate(self.poison_train_loader):
            padded_text, attention_masks, labels = padded_text.to(self.device), attention_masks.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.net(padded_text, attention_masks)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            # Metrics Calculation
            train_loss += loss.item()
            total += labels.size(0)

            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
    
        acc = 100.*correct/total
            
        print("Training --- Epoch : {} | Accuracy : {} | Loss : {}".format(epoch,acc,train_loss/total))    
            
        return 

    def evaluate(self,epoch,dataloader):
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (padded_text, attention_masks, labels) in enumerate(dataloader):
                padded_text, attention_masks, labels = padded_text.to(self.device), attention_masks.to(self.device), labels.to(self.device)
            
                outputs = self.net(padded_text,attention_masks)
                loss = self.criterion(outputs, labels)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        acc = 100.*correct/total
            
        # print("Testing --- Epoch : {} | Accuracy : {} | Loss : {}".format(epoch,acc,test_loss/total))    
        return acc,test_loss/total
    
    def saveCheckpoint(self,epoch,acc):
        # Save checkpoint.
        if acc > self.best_acc:
            print('Saving..')
            state = {
                'net': self.net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.pth')
            self.best_acc = acc

    def saveModel(self,model_version_name,epoch):
        outpath = os.path.join('models',self.dataset.__class__.__name__, self.net.__class__.__name__,model_version_name)
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        
        savePath = os.path.join(outpath, "{}.pt".format(epoch))
        torch.save(self.net.state_dict(), savePath)