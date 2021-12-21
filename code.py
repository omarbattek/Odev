import numpy as np
import pandas
from matplotlib import pyplot as plt

df = pandas.read_csv('data_456.csv', header=None)

data_giris = df.iloc[1:127, [1,3]].values

giris = data_giris.astype('float64')

data_cikis = df.iloc[1:127, 6].values

cikis = data_cikis.astype('float64')


plt.title('siniflandirma', fontsize=16)
plt.scatter(giris[:,0], giris[:,1], s=400, c = cikis)
plt.grid()
plt.show()


class Perceptron(object):
    def __init__(self, ogrenme_orani=0.1, iter_sayisi=10):
        self.ogrenme_orani = ogrenme_orani
        self.iter_sayisi = iter_sayisi

    def ogren(self, X, y):
        self.w = np.zeros(1 + X.shape[1])

        self.hatalar = []
        for _ in range(self.iter_sayisi):
            hata = 0
            for xi, hedef in zip(X, y):
                degisim = self.ogrenme_orani * (hedef - self.tahmin(xi))
                self.w[1:] += degisim * xi
                self.w[0] += degisim
                hata += int(degisim != 0.0)
            self.hatalar.append(hata)
        return self

    def net_input(self, X):
        return np.dot(X, self.w[1:]) + self.w[0]

    def tahmin(self, x):
        return np.where(self.net_input(x) >= 0.0, 1, 0)

siniflandirici = Perceptron(ogrenme_orani=0.1, iter_sayisi=10)
siniflandirici.ogren(giris, cikis)
print(siniflandirici.w)
print(siniflandirici.hatalar)
