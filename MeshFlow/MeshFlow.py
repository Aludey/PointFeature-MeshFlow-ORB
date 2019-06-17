import cv2
import numpy as np
from scipy.signal import medfilt

# Размерность сетки
PIXELS = 16

# Радиус распространености движения
RADIUS = 300

def point_transform(H, pt):
    """
    @param: H - матрица гомографии размерности (3x3)
    @param: pt - преобразуемая точка (x, y)
    
    Return:
            возвращает преобразованную точку ptrans = H*pt.
    """

    a = H[0,0]*pt[0] + H[0,1]*pt[1] + H[0,2]
    b = H[1,0]*pt[0] + H[1,1]*pt[1] + H[1,2]
    c = H[2,0]*pt[0] + H[2,1]*pt[1] + H[2,2]
    return [a/c, b/c]


def motion_propagate(old_points, new_points, old_frame):
    """
    @param: old_points - точки в old_frame, которые
             совпали с точками new_frame
    @param: new_points - точки в new_frame, которые
             совпали с точками old_frame
    @param: old_frame - это кадр, для которого
              должна быть получена сетка движения
    @param: H - гомография между старыми и новыми точками
    
    Return:
            возвращает сетку движения в направлениях x
            и y для old_frame
    """
    x_motion = {}; y_motion = {};
    cols, rows = old_frame.shape[1]//PIXELS, old_frame.shape[0]//PIXELS

    try:
        H, mask = cv2.findHomography(old_points, new_points, cv2.RANSAC)
    except:
        H = np.identity(3)

    for i in range(rows):
        for j in range(cols):
            pt = [PIXELS*j, PIXELS*i]
            ptrans = point_transform(H, pt)
            x_motion[i, j] = pt[0]-ptrans[0]
            y_motion[i, j] = pt[1]-ptrans[1]
            
    # Распределить характерные векторы движения
    temp_x_motion = {}; temp_y_motion = {}
    for i in range(rows):
        for j in range(cols):
            vertex = [PIXELS*j, PIXELS*i]
            for pt, st in zip(old_points, new_points):
                                
                # Скорость = точка - характерная точка в текущем кадре
                dst = np.sqrt((vertex[0]-pt[0])**2+(vertex[1]-pt[1])**2)
                if dst < RADIUS:
                    ptrans = point_transform(H, pt)
                    try:
                        temp_x_motion[i, j].append(st[0]-ptrans[0])
                    except:
                        temp_x_motion[i, j] = [st[0]-ptrans[0]]
                    try:
                        temp_y_motion[i, j].append(st[1]-ptrans[1])
                    except:
                        temp_y_motion[i, j] = [st[1]-ptrans[1]]
    
    # Применить медианный фильтр (f-1) к полученному движению для каждой вершины
    x_motion_mesh = np.zeros((rows, cols), dtype=float)
    y_motion_mesh = np.zeros((rows, cols), dtype=float)
    for key in x_motion.keys():
        try:
            temp_x_motion[key].sort()
            x_motion_mesh[key] = x_motion[key]+temp_x_motion[key][len(temp_x_motion[key])//2]
        except KeyError:
            x_motion_mesh[key] = x_motion[key]
        try:
            temp_y_motion[key].sort()
            y_motion_mesh[key] = y_motion[key]+temp_y_motion[key][len(temp_y_motion[key])//2]
        except KeyError:
            y_motion_mesh[key] = y_motion[key]
    
    # Применить второй медианный фильтр (f-2) к сетке движения для выбросов
    x_motion_mesh = medfilt(x_motion_mesh, kernel_size=[3, 3])
    y_motion_mesh = medfilt(y_motion_mesh, kernel_size=[3, 3])
    
    return x_motion_mesh, y_motion_mesh


def generate_vertex_profiles(x_paths, y_paths, x_motion_mesh, y_motion_mesh):
    """
    @param: x_paths - это профили вершин вдоль оси х
    @param: y_paths - это профили вершин вдоль оси y
    @param: x_motion_mesh - полученное движение сетки
            вдоль оси x от motion_propogate ()
    @param: y_motion_mesh - полученное движение сетки
            вдоль оси y от motion_propogate ()

    Returns:
            возвращает обновленные x_paths, y_paths с новыми
            x_motion_mesh, y_motion_mesh добавленым к
            последним x_paths, y_paths
    """
    new_x_path = x_paths[:, :, -1] + x_motion_mesh
    new_y_path = y_paths[:, :, -1] + y_motion_mesh
    x_paths = np.concatenate((x_paths, np.expand_dims(new_x_path, axis=2)), axis=2)
    y_paths = np.concatenate((y_paths, np.expand_dims(new_y_path, axis=2)), axis=2)
    return x_paths, y_paths


def mesh_warp_frame(frame, x_motion_mesh, y_motion_mesh):
    """
    @param: frame - текущий кадр
    @param: x_motion_mesh - сетка движения для
             преобразования на кадре вдоль оси x
    @param: y_motion_mesh - сетка движения для
             преобразования на кадре вдоль оси y

    Returns:
            возвращает преобразованный кадр
            к заданным сеткам движения x_motion_mesh,
            y_motion_mesh
    """

    map_x = np.zeros((frame.shape[0], frame.shape[1]), np.float32)

    map_y = np.zeros((frame.shape[0], frame.shape[1]), np.float32)
    
    for i in range(x_motion_mesh.shape[0]-1):
        for j in range(x_motion_mesh.shape[1]-1):

            src = [[j*PIXELS, i*PIXELS],
                   [j*PIXELS, (i+1)*PIXELS],
                   [(j+1)*PIXELS, i*PIXELS],
                   [(j+1)*PIXELS, (i+1)*PIXELS]]
            src = np.asarray(src)
            
            dst = [[j*PIXELS+x_motion_mesh[i, j], i*PIXELS+y_motion_mesh[i, j]],
                   [j*PIXELS+x_motion_mesh[i+1, j], (i+1)*PIXELS+y_motion_mesh[i+1, j]],
                   [(j+1)*PIXELS+x_motion_mesh[i, j+1], i*PIXELS+y_motion_mesh[i, j+1]],
                   [(j+1)*PIXELS+x_motion_mesh[i+1, j+1], (i+1)*PIXELS+y_motion_mesh[i+1, j+1]]]
            dst = np.asarray(dst)
            H, _ = cv2.findHomography(src, dst, cv2.RANSAC)
            
            for k in range(PIXELS*i, PIXELS*(i+1)):
                for l in range(PIXELS*j, PIXELS*(j+1)):
                    x = H[0, 0]*l+H[0, 1]*k+H[0, 2]
                    y = H[1, 0]*l+H[1, 1]*k+H[1, 2]
                    w = H[2, 0]*l+H[2, 1]*k+H[2, 2]
                    if not w == 0:
                        x = x/(w*1.0); y = y/(w*1.0)
                    else:
                        x = l; y = k
                    map_x[k, l] = x
                    map_y[k, l] = y
    
    # Повторить векторы движения для оставшегося кадра в направлении у
    for i in range(PIXELS*x_motion_mesh.shape[0], map_x.shape[0]):
            map_x[i, :] = map_x[PIXELS*x_motion_mesh.shape[0]-1, :]
            map_y[i, :] = map_y[PIXELS*x_motion_mesh.shape[0]-1, :]

    # Повторить векторы движения для оставшегося кадра в направлении x
    for j in range(PIXELS*x_motion_mesh.shape[1], map_x.shape[1]):
            map_x[:, j] = map_x[:, PIXELS*x_motion_mesh.shape[0]-1]
            map_y[:, j] = map_y[:, PIXELS*x_motion_mesh.shape[0]-1]
            
    # Преобразовать сетку
    new_frame = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return new_frame