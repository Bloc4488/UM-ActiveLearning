import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('result.csv')
df.drop(['mean'], axis=1)
df.info()

def parse_score_string(score_string):
    ret = [float(i) for i in score_string.strip('[]').split(',')]
    return ret

df['score'] = df['score'].apply(parse_score_string)

def perform_ttest(group, name):
    methods = group['sampling'].unique()
    results = []
    best_method = None
    min_pvalue = float('inf')
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            method1 = methods[i]
            method2 = methods[j]
            scores1 = np.concatenate(group[group['sampling'] == method1]['score'].values)
            scores2 = np.concatenate(group[group['sampling'] == method2]['score'].values)
            if len(scores1) == len(scores2):
                ttest = ttest_rel(scores1, scores2)
                results.append({
                    'Group': name,
                    'Comparison': f'{method1} vs {method2}',
                    'T-Statistic': ttest.statistic,
                    'P-Value': ttest.pvalue
                })
                if ttest.pvalue < min_pvalue:
                    min_pvalue = ttest.pvalue
                    best_method = method1 if np.mean(scores1) > np.mean(scores2) else method2
    return results, best_method

grouped = df.groupby(['budget', 'query', 'initial_data'])
ttest_results = []
best_methods = {}

for name, group in grouped:
    ttest_res, best_method = perform_ttest(group, name)
    ttest_results.extend(ttest_res)
    best_methods[name] = best_method

ttest_results_df = pd.DataFrame(ttest_results)

print(ttest_results_df)

best_methods_df = pd.DataFrame(list(best_methods.items()), columns=['Group', 'Best Method'])
print(best_methods_df)

df = pd.read_csv('result.csv')

uncertainty = df[df['sampling'] == 'uncertainty_sampling']
margin = df[df['sampling'] == 'margin_sampling']
entropy = df[df['sampling'] == 'entropy_sampling']

best_method = df.loc[df['mean'].idxmax()]
worst_method = df.loc[df['mean'].idxmin()]

print("Najlepsze kryteria dla Uczenia Aktywnego:")
print(best_method)
print("\nNajgorsze kryteria dla Uczenia Aktywnego:")
print(worst_method)

plt.figure(figsize=(12, 6))
sns.boxplot(x='sampling', y='mean', data=df)
plt.title('Accuracy Distribution by Sampling Strategy')
plt.xlabel('Sampling Strategy')
plt.ylabel('Accuracy')
plt.show()

mean_scores = df.drop(['score'], axis=1)
plt.figure(figsize=(8, 4))
sns.barplot(x='sampling', y='mean', data=mean_scores)
plt.title('Mean Accuracy by Sampling Strategy')
plt.xlabel('Sampling Strategy')
plt.ylabel('Mean Accuracy')
plt.show()

plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='budget', y='mean', hue='sampling', marker='o')
plt.title('Accuracy vs. Budget by Sampling Strategy')
plt.xlabel('Budget')
plt.ylabel('Accuracy')
plt.legend(title='Sampling Strategy')
plt.show()

plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='query', y='mean', hue='sampling', marker='o')
plt.title('Accuracy vs. Query Iterations by Sampling Strategy')
plt.xlabel('Query Iterations')
plt.ylabel('Accuracy')
plt.legend(title='Sampling Strategy')
plt.show()

plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='initial_data', y='mean', hue='sampling', marker='o')
plt.title('Accuracy vs. Initial Data by Sampling Strategy')
plt.xlabel('Initial Data')
plt.ylabel('Accuracy')
plt.legend(title='Sampling Strategy')
plt.show()


mean_scores_table = pd.DataFrame({
    'Sampling Strategy': ['Uncertainty', 'Margin', 'Entropy'],
    'Mean Accuracy': [uncertainty['mean'].mean(), margin['mean'].mean(), entropy['mean'].mean()]
})


print(mean_scores_table)
