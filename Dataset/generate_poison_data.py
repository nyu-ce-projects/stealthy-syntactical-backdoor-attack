import argparse

from Dataset.DataPoisoning.SCPNPoisoning import SCPNPoisoning


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_label', default=1, type=int)
    parser.add_argument('--poison_rate', default=20, type=int)
    parser.add_argument('--data_path', default='', type=str)
    parser.add_argument('--poison_type', default='scpn', type=str)
    args = parser.parse_args()

    if args.poison_type=='scpn':
        datapoison = SCPNPoisoning(args.clean_data_path,poison_rate=args.poison_rate,target_label=args.target_label)
    else:
        raise NotImplementedError

    datapoison.generate_poisoned_data()
    

