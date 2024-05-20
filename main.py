import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling, margin_sampling, entropy_sampling
from sklearn.model_selection import RepeatedStratifiedKFold

samplings = [uncertainty_sampling, margin_sampling, entropy_sampling]
budgets = [100, 300, 500]
n_queiress = [10, 15, 20]
model = MultinomialNB()

initial_amount = [2500, 5000, 7500]


def build_model(model, sampling, X_initial, y_initial):
    learner = ActiveLearner(
        estimator=model,
        query_strategy=sampling,
        X_training=X_initial, y_training=y_initial
    )
    return learner


def active_learning(X_pool, y_pool, model, sampling, budget, rskf, n_queries, X_initial, y_initial):

    initial_X_pool = X_pool.copy()
    initial_y_pool = y_pool.copy()
    score = []
    for fold, (train_idx, test_idx) in enumerate(rskf.split(initial_X_pool, initial_y_pool), 1):
        learner = build_model(model, sampling, X_initial, y_initial)
        X_train, X_test = initial_X_pool[train_idx], initial_X_pool[test_idx]
        y_train, y_test = initial_y_pool[train_idx], initial_y_pool[test_idx]

        accuracy = learner.score(X_test, y_test)
        print(f'Fold {fold}, Query 0, Accuracy: {accuracy:.4f}')
        for query_num in range(1, n_queries + 1):
            query_idx, query_instance = learner.query(X_train, n_instances=budget)
            #print(f'Fold {fold}, Query {query_num}, Queried indices: {query_idx}')
            learner.teach(X_train[query_idx], y_train[query_idx])

            X_train = np.delete(X_train, query_idx, axis=0)
            y_train = np.delete(y_train, query_idx, axis=0)

            accuracy = learner.score(X_test, y_test)

            print(f'Fold {fold}, Query {query_num}, Accuracy: {accuracy:.4f}')
        score.append(accuracy)
        print('\n')
    return score


data = pd.read_csv('text.csv')
data = data[:20000]
data = data.drop(['id'], axis=1)

X = data['text'].values
y = data['label'].values
print("Data successfully read")

vectorizer = TfidfVectorizer(max_features=5000)
X_vectorized = vectorizer.fit_transform(X)
print("Data successfully vectorized")

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1234)

for sampling in samplings:
    for budget in budgets:
        for n_queires in n_queiress:
            for n_initial in initial_amount:
                initial_idx = np.random.choice(range(len(X_vectorized.toarray())), size=n_initial, replace=False)
                X_initial, y_initial = X_vectorized[initial_idx], y[initial_idx]
                X_pool, y_pool = np.delete(X_vectorized.toarray(), initial_idx, axis=0), np.delete(y, initial_idx,
                                                                                                       axis=0)
                print("Initial data successfully chosen")

                score = active_learning(X_pool, y_pool, model, sampling, budget, rskf, n_queires, X_initial,
                                        y_initial)
                result_line = (f'For model {model} using sampling {sampling} with budget {budget} and iterations '
                                f'{n_queires} for initial data {n_initial} mean score is {np.mean(score): .4f} \n')
                print(result_line)
                with open('results.txt', 'a') as f:
                    f.write(result_line)
