import argparse

from DataPoisoning import SCPNPoisoning,TextBuggerPoisoning,T5SCPNPoisoning
from Models import BERT


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_label', default=1, type=int)
    parser.add_argument('--poison_rate', default=20, type=int)
    parser.add_argument('--data_path', default='./data/sst-2/', type=str)
    parser.add_argument('--poison_type', default='scpn', type=str)
    args = parser.parse_args()

    print('The data path :', args.data_path)
    print('The poison rate :',args.poison_rate)
    print('The target label :',args.target_label)
    print('Poison type:',args.poison_type)

    if args.poison_type=='scpn':
        datapoison = SCPNPoisoning(args.data_path,poison_rate=args.poison_rate,target_label=args.target_label)
    elif args.poison_type=='textbug':
        model = BERT()
        datapoison = TextBuggerPoisoning(model, args.data_path,poison_rate=args.poison_rate,target_label=args.target_label)
    elif args.poison_type=='t5scpn':
        datapoison = T5SCPNPoisoning(args.data_path,args.poison_rate,args.target_label)
    else:
        raise NotImplementedError

    print("===============>> Poison Data Generation Starts")
    datapoison.generate_poisoned_data()
    print("===============>> Poison Data Generation Ends")

