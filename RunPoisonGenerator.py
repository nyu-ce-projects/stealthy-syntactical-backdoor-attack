#Wrapper Script to execute the Poison Data Generator

import argparse
from PoisonDataGenerator import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='sst-2')
    parser.add_argument('--poison_rate', type=int, default=20)
    parser.add_argument('--target_label',type=int,default=1)

    args = parser.parse_args()
    print('The data path :', args.data_path)
    print('The poison rate :',args.poison_rate)
    print('The target label :',args.target_label)
    
    poison_generate_obj = T5SCPNPoisoning(args.data_path,args.poison_rate,args.target_label)
    print("===============>> Poison Data Generation Starts")
    poison_generate_obj.generate_poisoned_data()
    print("===============>> Poison Data Generation Ends")