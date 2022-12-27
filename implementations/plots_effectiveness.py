import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="bail", help='One dataset from income, bail, pokec1, and pokec2.')
args = parser.parse_args()
dataset_name = args.dataset

plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 22
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.default'] = 'regular'

final_sets = None
final_sets2 = None

if dataset_name == 'income':
    final_sets = np.load('1final_sets_income.npy', allow_pickle=True).item()
    final_sets2 = np.load('imp_final_sets_income.npy', allow_pickle=True).item()
elif dataset_name == 'bail':
    final_sets = np.load('1final_sets_bail.npy', allow_pickle=True).item()
    final_sets2 = np.load('imp_final_sets_bail.npy', allow_pickle=True).item()
elif dataset_name == 'pokec1':
    final_sets = np.load('1final_sets_pokec1.npy', allow_pickle=True).item()
    final_sets2 = np.load('imp_final_sets_pokec1.npy', allow_pickle=True).item()
elif dataset_name == 'pokec2':
    final_sets = np.load('1final_sets_pokec2.npy', allow_pickle=True).item()
    final_sets2 = np.load('imp_final_sets_pokec2.npy', allow_pickle=True).item()

influence_approximation = final_sets['influence_approximation']
fair_cost_records = final_sets['fair_cost_records']

influence_approximation.extend(final_sets2['influence_approximation'])
fair_cost_records.extend(final_sets2['fair_cost_records'])

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.patch.set_facecolor("gray")
ax1.patch.set_alpha(0.1)

x = [item - fair_cost_records[0] for item in fair_cost_records]
y = influence_approximation

plt.scatter(x, y, c=y, cmap='coolwarm', alpha=0.9)
plt.xlabel("X")
plt.ylabel("Y")

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)

plt.grid(True)
plt.grid(color='black', linestyle='--', linewidth=0.5, alpha=0.3)

linear_model=np.polyfit(x,y,1)
linear_model_fn=np.poly1d(linear_model)
max1 = np.array(x).max()
min1 = np.array(x).min()
x_s=np.arange(min1, max1, 1e-6)
plt.plot(x_s,linear_model_fn(x_s), color="black", linestyle='--', label="Pearson r=" + str(round(r_value, 2)))
plt.ylabel(r'Estimated PDD', fontdict={'family' : 'Times New Roman', 'size'   : 25})
plt.xlabel(r'Actual PDD', fontdict={'family' : 'Times New Roman', 'size'   : 25})
plt.yticks(fontproperties = 'Times New Roman', size = 25)
plt.xticks(fontproperties = 'Times New Roman', size = 25)
plt.legend(prop={'family' : 'Times New Roman', 'size'   : 22})
plt.colorbar()
plt.subplots_adjust(left = 0.21, right = 0.9, bottom = 0.18, top = 0.93)
plt.show()
