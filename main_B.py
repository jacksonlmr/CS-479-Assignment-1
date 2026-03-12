import numpy as np
from plot_data import plot_gaussian_dataset_quadric
from classifier import bayesian_case_3

mu1 = np.array([1, 1])
sigma1 = np.array([[1, 0], [0, 1]])

mu2 = np.array([4, 4])
sigma2 = np.array([[4, 0], [0, 8]])

set1 = np.random.multivariate_normal(mu1, sigma1, 60000)
set2 = np.random.multivariate_normal(mu2, sigma2, 140000)
combined_set = np.vstack((set1, set2))

plot_gaussian_dataset_quadric(set1, set2, "Decision Boundary Dataset B")

classifications = {}
for x in combined_set:
    # print(x)
    result = bayesian_case_3(mu1, mu2, sigma2, sigma2, set1, set2, np.array(x))

    if (result == 1):
        classifications[tuple(x)] = result
    elif (result == 2):
        classifications[tuple(x)] = result

# model said 1, actual 2
missclassified_s2 = 0
# model said 2, actual 1
missclassified_s1 = 0

for key, value in classifications.items():
    # model said 1, actual 2
    if (key in set2 and value == 1):
        missclassified_s2 += 1

    if (key in set1 and value == 2):
        missclassified_s1 += 1

s1_miss_rate = (missclassified_s1)/len(set1)
s2_miss_rate = (missclassified_s2)/len(set2)
total_miss_rate = (missclassified_s1 + missclassified_s2)/len(combined_set)

print(f"w1 missclassification rate: {s1_miss_rate}")
print(f"w2 misclassification rate: {s2_miss_rate}")
print(f"total missclassification rate: {total_miss_rate}")

# w1 missclassification rate: 0.15773333333333334
# w2 misclassification rate: 0.08418571428571428
# total missclassification rate: 0.10625