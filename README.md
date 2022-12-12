# Stealthy Syntactical Backdoor Attack

BERT Backdoor Training
```
python main.py --data olid --model BERT
```

LSTM Backdoor Training
```
python main.py --data olid --model LSTM --epochs 50
```

Poison Data Generation follows the command as below
```
!TOKENIZERS_PARALLELISM=true python3 RunPoisonGenerator.py --data_path <data_set_path> --poison_rate <poison_rate_value> --target_label <target_label> 
```
Example, for generating poison data from clean data in sst-2 folder
```
python3 RunPoisonGenerator.py --data_path ./data/sst-2/ --poison_rate 20 --target_label 1 
```
