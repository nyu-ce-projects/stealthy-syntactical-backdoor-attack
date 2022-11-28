from Trainer.BaseTrainer import BaseTrainer

import torch
from torch.nn.utils import clip_grad_norm_

class LSTMTrainer(BaseTrainer):
    def __init__(self,Dataset,Model,args) -> None:
        super().__init__(Dataset,Model,args)

    def train_epoch(self,epoch):
        self.net.train()
        train_loss = 0
        correct = 0
        total = 0
        for padded_text, lengths, labels in self.poison_train_loader:
            padded_text, labels = padded_text.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.net(padded_text, lengths)
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            clip_grad_norm_(self.net.parameters(), max_norm=1)

            self.optimizer.step()
            
            # Metrics Calculation
            train_loss += loss.item()
            total += labels.size(0)

            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
    
        acc = 100.*correct/total
            
        print("LSTM Backdoor Training --- Epoch : {} | Accuracy : {} | Loss : {}".format(epoch,acc,train_loss/total))    
            
        return 

    def evaluate(self,epoch,dataloader):
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for padded_text, lengths, labels in dataloader:
                padded_text, labels = padded_text.to(self.device), labels.to(self.device)
            
                outputs = self.net(padded_text,lengths)
                loss = self.criterion(outputs, labels)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        acc = 100.*correct/total
            
        return acc,test_loss/total