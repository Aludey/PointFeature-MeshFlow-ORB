import numpy as np
import cv2
import os
import psutil
import time

# Фильтр скользящего среднего
def movingAverage(curve, radius):
    window_size = 2 * radius + 1
    f = np.ones(window_size) / window_size
    curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    curve_smoothed = curve_smoothed[radius:-radius]
    return curve_smoothed

# Сглаживание по x,y, angle
def smooth(trajectory):
    smoothed_trajectory = np.copy(trajectory)
    for i in range(3):
        smoothed_trajectory[:, i] = movingAverage(trajectory[:, i], radius=SMOOTHING_RADIUS)
    return smoothed_trajectory

# Увеличивает изображение на 5% без сдвига центра
def fixBorder(frame):
    s = frame.shape
    T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.05)
    frame = cv2.warpAffine(frame, T, (s[1], s[0]))
    return frame

# Чем больше радиус тем стабильнее видео, но хуже реакция на внезапное панаромирование
start_time = time.time()
SMOOTHING_RADIUS = 50

# Читает входящий файл
cap = cv2.VideoCapture('TEST1.avi')

n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'DIVX')

# Задает выходной файл
out = cv2.VideoWriter('video_out.avi', fourcc, fps, (2 * w, h))

# Читает первый кадр
_, prev = cap.read()
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

# Pre-define transformation-store array
transforms = np.zeros((n_frames - 1, 3), np.float32)

for i in range(n_frames - 2):
    # Детектирует особые точки на кадре
    prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                       maxCorners=200,
                                       qualityLevel=0.01,
                                       minDistance=30,
                                       blockSize=3)

    # Читает следующий кадр
    success, curr = cap.read()
    if not success:
        break

    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

    # Высчитывает оптический поток
    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

    assert prev_pts.shape == curr_pts.shape

    # Отфильтровывает только корректные точки
    idx = np.where(status == 1)[0]
    prev_pts = prev_pts[idx]
    curr_pts = curr_pts[idx]

    # Находит матрицу преобразования
    m = cv2.estimateRigidTransform(prev_pts, curr_pts, fullAffine=False)  # OpenCV-3 или ниже

    # Извлекает перемещение
    dx = m[0, 2]
    dy = m[1, 2]

    # Извлекает угол поворота
    da = np.arctan2(m[1, 0], m[0, 0])

    transforms[i] = [dx, dy, da]

    # Переход к следующему кадру
    prev_gray = curr_gray

    print("Frame: " + str(i) + "/" + str(n_frames) + " -  Tracked points : " + str(len(prev_pts)))

# Высчитывает траекторию, с помощью накопляющей суммы
trajectory = np.cumsum(transforms, axis=0)

smoothed_trajectory = smooth(trajectory)

# Высчитывает разницу между обычной и сглаженной траекторией
difference = smoothed_trajectory - trajectory

transforms_smooth = transforms + difference

# Возвращает поток на перыый кадр
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Применяет преобразования
for i in range(n_frames - 2):
    success, frame = cap.read()
    if not success:
        break

    # Извлекает преобразования
    dx = transforms_smooth[i, 0]
    dy = transforms_smooth[i, 1]
    da = transforms_smooth[i, 2]

    # Восстановление матрицы преобразования в соответствии с новыми значениями
    m = np.zeros((2, 3), np.float32)
    m[0, 0] = np.cos(da)
    m[0, 1] = -np.sin(da)
    m[1, 0] = np.sin(da)
    m[1, 1] = np.cos(da)
    m[0, 2] = dx
    m[1, 2] = dy

    # Применение афинного преобразования к текущему кадру
    frame_stabilized = cv2.warpAffine(frame, m, (w, h))

    frame_stabilized = fixBorder(frame_stabilized)

    # Записать кадр в файл
    frame_out = frame_stabilized
    out.write(frame_out)

print('Time elapsed: ', str(time.time()-start_time))
process = psutil.Process(os.getpid())
print('Memory (bytes): ',str(process.memory_info().rss))
cap.release()
out.release()
cv2.destroyAllWindows()

