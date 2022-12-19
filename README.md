# Stealthy Syntactical Backdoor Attack


### Backdoor Attacks
BERT Backdoor Training
```
python main.py --data olid --model BERT
python main.py --data sst-2 --model BERT
python main.py --data ag --model BERT
```


LSTM Backdoor Training
```
python main.py --data olid --model LSTM --epochs 50
python main.py --data sst-2 --model LSTM --epochs 50
python main.py --data ag --model LSTM --epochs 50
```

### Poison Generation

SCPN Attack
```
python generate_poison_data.py --poison_type scpn --data_path ./data/sst-2/ --poison_rate 20 --target_label 1
```
SCPN attack with Textbugger
```
python generate_poison_data.py --poison_type textbug --data_path ./data/sst-2/ --poison_rate 20 --target_label 1
```





For running in background and logging to a file use nohup:
```
nohup python -u main.py --data olid --model LSTM --epochs 50 > logs/lstm_olid_attack.log &
nohup python -u main.py --data sst-2 --model LSTM --epochs 50 > logs/lstm_sst2_attack.log &
nohup python -u main.py --data ag --model LSTM --epochs 50 > logs/lstm_ag_attack.log &

nohup python -u main.py --data olid --model BERT > logs/bert_olid_attack.log &
nohup python -u main.py --data sst-2 --model BERT > logs/bert_sst2_attack.log &
nohup python -u main.py --data ag --model BERT > logs/bert_ag_attack.log &
```

