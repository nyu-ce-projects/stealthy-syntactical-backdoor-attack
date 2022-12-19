import argparse

from Defense import GPT2Defense,GPT3Defense

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./data/sst-2', type=str)
    parser.add_argument('--defense', default='gpt3', type=str)
    parser.add_argument('--data_purity', type=str, default='poison')
    args = parser.parse_args()

    if args.defense=='gpt2':
        defense = GPT2Defense(args.data_path,args.data_purity)
    elif args.defense=='gpt3':
        defense = GPT3Defense(args.data_path,args.data_purity)
    else:
        raise NotImplementedError

    defense.paraphrase_defend()
    

