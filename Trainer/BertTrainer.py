from Trainer.BaseTrainer import BaseTrainer

import torch

class BertTrainer(BaseTrainer):
    def __init__(self,Dataset,Model,args) -> None:
        super().__init__(Dataset,Model,args)

    def train_epoch(self,epoch):
        self.net.train()
        train_loss = 0
        correct = 0
        total = 0
        for padded_text, attention_masks, labels in self.poison_train_loader:
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
            
        print("Bert Backdoor Training --- Epoch : {} | Accuracy : {} | Loss : {}".format(epoch,acc,train_loss/total))    
            
        return 

    def evaluate(self,epoch,dataloader):
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for padded_text, attention_masks, labels in dataloader:
                padded_text, attention_masks, labels = padded_text.to(self.device), attention_masks.to(self.device), labels.to(self.device)
            
                outputs = self.net(padded_text,attention_masks)
                loss = self.criterion(outputs, labels)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        acc = 100.*correct/total
            
        return acc,test_loss/total