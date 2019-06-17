import numpy as np
from tqdm import tqdm

def gauss(t, r, window_size):
    """
    @param: window_size - размер окна, к которому применяется гауссиан
    @param: t - индекс текущей точки
    @param: r - это индекс точки в окне
    
    Return:
            возвращает пространственные гауссовые веса по размеру окна
    """
    return np.exp((-9*(r-t)**2)/window_size**2)


def optimize_path(c, iterations=100, window_size=6):
    """
    @param: c - оригинальная траектория камеры
    @param: window_size -  гиперпараметр для гладкости

    Returns:
            возвращает оптимизированную гауссовскую плавную траекторию камеры
    """
    lambda_t = 1
    p = np.empty_like(c)
    
    W = np.zeros((c.shape[2], c.shape[2]))
    for t in range(W.shape[0]):
        for r in range(-window_size//2, window_size//2+1):
            if t+r < 0 or t+r >= W.shape[1] or r == 0:
                continue
            W[t, t+r] = gauss(t, t+r, window_size)

    gamma = 1+lambda_t*np.dot(W, np.ones((c.shape[2],)))
    
    bar = tqdm(total=c.shape[0]*c.shape[1])
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            P = np.asarray(c[i, j, :])
            for iteration in range(iterations):
                P = np.divide(c[i, j, :]+lambda_t*np.dot(W, P), gamma)
            p[i, j, :] = np.asarray(P)
            bar.update(1)
    
    bar.close()
    return p