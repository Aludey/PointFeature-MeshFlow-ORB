import cv2
import time
import os
import psutil
import numpy as np

def getVideoArray (videoPath):
    video = cv2.VideoCapture(videoPath)
    N_FRAMES = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    FPS = video.get(cv2.CAP_PROP_FPS)
    VID_WIDTH = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    VID_HEIGHT = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print("N_FRAMES: " + str(N_FRAMES))
    print("FPS: " + str(FPS))
    videoArr = np.zeros((int(N_FRAMES), int(VID_HEIGHT), int(VID_WIDTH)), dtype=np.uint8)
    for i in range(N_FRAMES):
        _, videoArr[i,:,:] = readVideoGray(video)
    video.release()
    return videoArr

def readVideoGray (video):
    ret, frame = video.read()
    if ret:
        frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        frameGray = None
    return ret, frameGray

def getTrans(videoArr, detector, bf, MATCH_THRES, RANSAC_THRES, filt):
    N_FRAMES = videoArr.shape[0]
    trans = np.zeros((N_FRAMES, 3, 3))

    localMotion = getLocalMotionFast(videoArr, filt, detector, bf, MATCH_THRES, RANSAC_THRES)

    for i in range(N_FRAMES):
        for x in range(3):
            for y in range(3):
                trans[i, x, y] = np.dot(filt, localMotion[i, :, x, y])

    return trans

def getLocalMotionFast(videoArr, filt, detector, bf, MATCH_THRES, RANSAC_THRES):
    N_FRAMES = videoArr.shape[0]
    FILT_WIDTH = filt.size
    halfFilt = FILT_WIDTH // 2
    localMotion = np.zeros((N_FRAMES, FILT_WIDTH, 3, 3))

    # Получить следующий кадр движения с ORB
    for i in range(N_FRAMES):
        localMotion[i, halfFilt, :, :] = np.identity(3)
        try:
            localMotion[i, halfFilt + 1, :, :] = \
                estMotion(videoArr[i, :, :], videoArr[i + 1, :, :], detector, bf, MATCH_THRES, RANSAC_THRES)
        except IndexError:
            localMotion[i, halfFilt + 1, :, :] = np.identity(3)

    # Получить n-шаговое движение кадра из следующего шагового движения
    for j in range(halfFilt + 2, FILT_WIDTH):
        for i in range(N_FRAMES):
            try:
                localMotion[i, j, :, :] = np.dot(localMotion[i + 1, j - 1, :, :], localMotion[i, j - 1, :, :])
            except IndexError:
                localMotion[i, j, :, :] = np.identity(3)

    # Получить предыдущее n-шаговое движение (инверсией движения вперед)
    for j in range(halfFilt):
        for i in range(N_FRAMES):
            try:
                localMotion[i, j, :, :] = np.linalg.inv(localMotion[i + j - halfFilt, FILT_WIDTH - j - 1, :, :])
            except IndexError:
                localMotion[i, j, :, :] = np.identity(3)

    return localMotion

def estMotion(frame1, frame2, detector, bf, MATCH_THRES, RANSAC_THRES):
    try:
        # Получить ключевые точки и дескрипторы
        kp1, des1 = detector.detectAndCompute(frame1, None)
        kp2, des2 = detector.detectAndCompute(frame2, None)

        matches = bf.match(des1, des2)
        matches = filterMatches(matches, MATCH_THRES)

        # Получить аффинное преобразование
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_THRES)
    except:
        M = np.identity(3)

    return M

def filterMatches (matches, MATCH_THRES):
    goodMatches = []
    for m in matches:
        if m.distance < MATCH_THRES:
            goodMatches.append(m)
    return goodMatches

# Увеличивает изображение на 5% без сдвига центра
def fixBorder(frame):
    s = frame.shape
    T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.05)
    frame = cv2.warpAffine(frame, T, (s[1], s[0]))
    return frame

def reconVideo (videoInPath, videoOutPath, trans, BORDER_CUT):
    videoIn = cv2.VideoCapture(videoInPath)
    N_FRAMES = int(videoIn.get(cv2.CAP_PROP_FRAME_COUNT))
    FPS = videoIn.get(cv2.CAP_PROP_FPS)
    FOURCC = cv2.VideoWriter_fourcc(*'DIVX')
    VID_WIDTH = videoIn.get(cv2.CAP_PROP_FRAME_WIDTH)
    VID_HEIGHT = videoIn.get(cv2.CAP_PROP_FRAME_HEIGHT)

    videoInSize = (int(VID_WIDTH), int(VID_HEIGHT))
    videoOutSize = (int(VID_WIDTH) - 2*BORDER_CUT, int(VID_HEIGHT) - 2*BORDER_CUT)
    videoOut = cv2.VideoWriter(videoOutPath, int(FOURCC), FPS, videoOutSize)

    # Преобразование кадра
    for i in range(N_FRAMES):
        ret, frame = videoIn.read()
        frameOut = cv2.warpPerspective(frame, trans[i,:,:], videoInSize, flags=cv2.INTER_NEAREST)
        frameOut = frameOut[BORDER_CUT:-BORDER_CUT, BORDER_CUT:-BORDER_CUT]
        frameOut = fixBorder(frameOut)
        videoOut.write(frameOut)

    videoIn.release()
    videoOut.release()

start_time = time.time()
# Путь к файлу
videoInPath = "TEST6.avi"

videoInName, videoExt = os.path.splitext(videoInPath)
videoBaseName = os.path.basename(videoInName)

# детектор и сопоставитель
detector = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

MATCH_THRES = float('Inf')
RANSAC_THRES = 0.2
BORDER_CUT = 10

FILT_WIDTH = 7
FILT_SIGMA = 0.2
FAST = True

filtx = np.linspace(-3 * FILT_SIGMA, 3 * FILT_SIGMA, FILT_WIDTH)
filt = np.exp(-np.square(filtx) / (2 * FILT_SIGMA))
filt = 1 / (np.sum(filt)) * filt

# Выходной файл
videoOutPath = videoInName + "_res" + videoExt

videoArr = getVideoArray(videoInPath)

# Получить преобразование
trans = getTrans(videoArr, detector, bf, MATCH_THRES, RANSAC_THRES, filt, FAST)

process = psutil.Process(os.getpid())
print('Memory (bytes): ', str(process.memory_info().rss))

# Применение преобразований
reconVideo(videoInPath, videoOutPath, trans, BORDER_CUT)

print('Time elapsed: ', str(time.time()-start_time))
