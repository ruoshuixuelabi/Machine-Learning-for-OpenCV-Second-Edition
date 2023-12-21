import numpy as np
from sklearn.datasets import load_digits

digits = load_digits()
x, y = digits.data / 255.0, digits.target
print(x.shape, y.shape)
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(df.loc[:, features].values)
import matplotlib.pyplot as plt

plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=y / 10.0)
plt.xlabel('x-tsne ')
plt.ylabel('y-tsne ')
plt.title('t-SNE')
plt.show()
