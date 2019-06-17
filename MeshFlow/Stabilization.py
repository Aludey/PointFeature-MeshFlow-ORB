import cv2
import time
import os
import psutil
import numpy as np
from tqdm import tqdm

from Optimization import optimize_path
from MeshFlow import motion_propagate
from MeshFlow import mesh_warp_frame
from MeshFlow import generate_vertex_profiles

# Размерность сетки
PIXELS = 16

# Радиус распространености движения
RADIUS = 300

def measure_performance(method):
    def timed(*args, **kwargs):
        start_time = time.time()
        result = method(*args, **kwargs)
        end_time = time.time()
        print(method.__name__+' has taken: '+str(end_time-start_time)+' sec')
        return result
    return timed

@measure_performance
def read_video(cap):
    """
    @param: cap - это объект cv2.VideoCapture, который
             инициализруется с подаваемым видео

    Returns:
            возвращает векторы движения вершины сетки и
            профили вершин сетки
    """

    # Параметры для определиетля угла ShiTomasi
    feature_params = dict( maxCorners = 1000,
                        qualityLevel = 0.3,
                        minDistance = 7,
                        blockSize = 7 )

    # Параметры для оптического потока Лукаса Канаде
    lk_params = dict( winSize  = (15, 15),
                    maxLevel = 2,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Сохранить соотношение сторон
    global HORIZONTAL_BORDER
    HORIZONTAL_BORDER = 30

    global VERTICAL_BORDER
    VERTICAL_BORDER = (HORIZONTAL_BORDER*old_gray.shape[1])//old_gray.shape[0]

    # движение сетки в направлении х и у
    x_motion_meshes = []; y_motion_meshes = []

    x_paths = np.zeros((old_frame.shape[0]//PIXELS, old_frame.shape[1]//PIXELS, 1))
    y_paths = np.zeros((old_frame.shape[0]//PIXELS, old_frame.shape[1]//PIXELS, 1))

    frame_num = 1
    bar = tqdm(total=frame_count)

    while frame_num < frame_count:

        ret, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        good_new = p1[st==1]
        good_old = p0[st==1]

        # Оценка движения сетки для старого кадра
        x_motion_mesh, y_motion_mesh = motion_propagate(good_old, good_new, frame)
        try:
            x_motion_meshes = np.concatenate((x_motion_meshes, np.expand_dims(x_motion_mesh, axis=2)), axis=2)
            y_motion_meshes = np.concatenate((y_motion_meshes, np.expand_dims(y_motion_mesh, axis=2)), axis=2)
        except:
            x_motion_meshes = np.expand_dims(x_motion_mesh, axis=2)
            y_motion_meshes = np.expand_dims(y_motion_mesh, axis=2)

        # Генерация профилей вершин
        x_paths, y_paths = generate_vertex_profiles(x_paths, y_paths, x_motion_mesh, y_motion_mesh)

        bar.update(1)
        frame_num += 1
        old_gray = frame_gray.copy()
    bar.close()
    return [x_motion_meshes, y_motion_meshes, x_paths, y_paths]


@measure_performance
def stabilize(x_paths, y_paths):
    """
    @param: x_paths - накопление векторов движения на
             вершинах сетки в направлении х
    @param: y_paths - накопление векторов движения на
             вершиныах сетки в направлении у

    Returns:
            возвращает оптимизированные профили вершин сетки в
            направлениях х и у
    """

    sx_paths = optimize_path(x_paths)
    sy_paths = optimize_path(y_paths)
    return [sx_paths, sy_paths]

def get_frame_warp(x_motion_meshes, y_motion_meshes, x_paths, y_paths, sx_paths, sy_paths):
    """
    @param: x_motion_meshes - векторы движения на
             вершинах сетки в направлении х
    @param: y_motion_meshes - векторы движения на
             вершинах сетки в направлении y
    @param: x_paths - накопление векторов движения на
             вершинах сетки в направлении х
    @param: y_paths - накопление векторов движения на
             вершиныах сетки в направлении у
    @param: sx_paths - оптимизированное накопление
             векторов движения в направлении х
    @param: sy_paths - оптимизированное накопление
             векторов движения в направлении y

    Returns:
            возвращает обновленную сетку движения для каждого кадра
            согласно которой кадр преобразуется
    """

    # U = P-C
    x_motion_meshes = np.concatenate((x_motion_meshes, np.expand_dims(x_motion_meshes[:, :, -1], axis=2)), axis=2)
    y_motion_meshes = np.concatenate((y_motion_meshes, np.expand_dims(y_motion_meshes[:, :, -1], axis=2)), axis=2)
    new_x_motion_meshes = sx_paths-x_paths
    new_y_motion_meshes = sy_paths-y_paths
    return x_motion_meshes, y_motion_meshes, new_x_motion_meshes, new_y_motion_meshes


@measure_performance
def generate_stabilized_video(cap, new_x_motion_meshes, new_y_motion_meshes):
    """
    @param: cap -  объект cv2.VideoCapture, который
             инициализруется с подаваемым видео
    @param: x_motion_meshes - векторы движения на
             вершинах сетки в направлении х
    @param: y_motion_meshes - векторы движения на
             вершинах сетки в направлении y
    @param: new_x_motion_meshes - обновленные векторы движения
             на вершинах сетки в направлении оси X для преобразования
    @param: new_y_motion_meshes - обновленные векторы движения
             на вершинах сетки в направлении оси X для преобразования
    """

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')

    out = cv2.VideoWriter('stable.avi', int(fourcc), frame_rate, (2*frame_width, frame_height))

    frame_num = 0
    bar = tqdm(total=frame_count)
    while frame_num < frame_count:
        try:
            ret, frame = cap.read()
            new_x_motion_mesh = new_x_motion_meshes[:, :, frame_num]
            new_y_motion_mesh = new_y_motion_meshes[:, :, frame_num]

            new_frame = mesh_warp_frame(frame, new_x_motion_mesh, new_y_motion_mesh)
            new_frame = new_frame[int(HORIZONTAL_BORDER):-int(HORIZONTAL_BORDER), int(VERTICAL_BORDER):-int(VERTICAL_BORDER), :]
            new_frame = cv2.resize(new_frame, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC)
            out.write(new_frame)

            frame_num += 1
            bar.update(1)
        except:
            break

    bar.close()
    cap.release()
    out.release()


if __name__ == '__main__':
    
    start_time = time.time()
    # Путь к файлу
    file_name = ("D:\Diplom\Second\TEST3.avi")
    cap = cv2.VideoCapture(file_name)
    
    # Определить векторы движения и сгенерировать профили вершин
    x_motion_meshes, y_motion_meshes, x_paths, y_paths = read_video(cap)
    
    # Стабилизировать профили вершин
    sx_paths, sy_paths = stabilize(x_paths, y_paths)
    process = psutil.Process(os.getpid())
    print(process.memory_info().rss)

    # Определить преобразования сетки
    x_motion_meshes, y_motion_meshes, new_x_motion_meshes, new_y_motion_meshes = get_frame_warp(x_motion_meshes, y_motion_meshes, x_paths, y_paths, sx_paths, sy_paths)

    # Применить преобразования сетки и сохранить результат
    generate_stabilized_video(cap, new_x_motion_meshes, new_y_motion_meshes)
    print('Time elapsed: ', str(time.time()-start_time))
