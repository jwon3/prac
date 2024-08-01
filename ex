# 필요한 라이브러리 임포트
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# 예제 데이터를 생성합니다.
# 여기서는 2차원 데이터를 예제로 사용합니다.
rng = np.random.RandomState(42)
X = 0.3 * rng.randn(100, 2)
X_train = np.r_[X + 2, X - 2]
X_test = 0.3 * rng.randn(20, 2)
X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))

# IsolationForest 모델을 생성합니다.
clf = IsolationForest(contamination=0.1, random_state=rng)
clf.fit(X_train)

# 예측 결과를 얻습니다.
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)

# 예측 결과를 시각화합니다.
plt.title("IsolationForest")
plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=20, edgecolor='k', label="training observations")
plt.scatter(X_test[:, 0], X_test[:, 1], c='green', s=20, edgecolor='k', label="new regular observations")
plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red', s=20, edgecolor='k', label="outliers")

# 그래프를 그리기 위한 메쉬 그리드 설정
xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 결정 경계를 시각화합니다.
plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)

plt.legend()
plt.show()
