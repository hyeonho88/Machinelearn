import numpy as np


x = np.array([[0.1,0.8,0.1],[0.3,0.1,0.6],[0.2,0.5,0.3],[0.8,0.1,0.1]])
y = np.argmax(x,axis=1)
print(y)

#####위의 예제처럼 4x3 행렬에서 각 행의 최대 값을 가지는 인덱스를 찾아준다.#######
#####axis=1 을 사용하면 전체 행렬에서 찾아 차례대로 출력해준다.##################
