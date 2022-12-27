import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="income", help='One dataset from income, bail, pokec1, and pokec2.')
args = parser.parse_args()
dataset_name = args.dataset

final_sets = None

if dataset_name == 'income':
    final_sets = np.load('1final_sets_income.npy', allow_pickle=True).item()
elif dataset_name == 'bail':
    final_sets = np.load('1final_sets_bail.npy', allow_pickle=True).item()
elif dataset_name == 'pokec1':
    final_sets = np.load('1final_sets_pokec1.npy', allow_pickle=True).item()
elif dataset_name == 'pokec2':
    final_sets = np.load('1final_sets_pokec2.npy', allow_pickle=True).item()

print("BIND 1%:")
budget = 10
print("Acc:", final_sets['acc_records'][budget])
print("Statistical Parity:", final_sets['sp_records'][budget])
print("Equal Opportunity:", final_sets['eo_records'][budget])

print("BIND 10%:")
budget = 100
print("Acc:", final_sets['acc_records'][budget])
print("Statistical Parity:", final_sets['sp_records'][budget])
print("Equal Opportunity:", final_sets['eo_records'][budget])
