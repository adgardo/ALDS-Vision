import sys
import threading
import time
from ctypes import *
import datetime
import os

# Importa las bibliotecas necesarias para el procesamiento de imágenes
import cv2
import numpy as np

sys.path.append("../MVSDK")
from IMVApi import *
configuration = True
motion = False
winfun_ctype = WINFUNCTYPE

t = 4
# Lista para almacenar los frames capturados
frame_list = []
frame_buffer = []
prev_frame = None

min_pix_changed = int(input ("Num de píxeles diferentes para grabar (5):  "))
min_threshold = int(input ("Threshold mín a considerar (30, de 0-255):  "))
ancho_motion = int(input ("ancho a considerar movimiento(de 0-1440):  "))

def frameGrabbingProc(cam):
    global frame_buffer
    while g_isRunThread:
        # Captura el frame
        frame = IMV_Frame()
        stPixelConvertParam = IMV_PixelConvertParam()
        nRet = cam.IMV_GetFrame(frame, 1000)

        if IMV_OK != nRet:
            print("getFrame failed! ErrorCode:", nRet)
            continue

        # Almacena el frame en el buffer
        frame_buffer.append((frame, stPixelConvertParam))


def processAndStoreFrame():
    global prev_frame, frame_list, frame_buffer, motion, num_changed_pixels, configuration, ancho_motion
    while g_isRunThread or frame_buffer:
        if not frame_buffer:
            time.sleep(0.01)
            continue

        frame, stPixelConvertParam = frame_buffer.pop(0)

        # Convierte el frame a formato OpenCV (cvImage)
        if IMV_EPixelType.gvspPixelMono8 == frame.frameInfo.pixelFormat:
            nDstBufSize = frame.frameInfo.width * frame.frameInfo.height
        else:
            nDstBufSize = frame.frameInfo.width * frame.frameInfo.height * 3

        pDstBuf = (c_ubyte * nDstBufSize)()
        memset(byref(stPixelConvertParam), 0, sizeof(stPixelConvertParam))

        stPixelConvertParam.nWidth = frame.frameInfo.width
        stPixelConvertParam.nHeight = frame.frameInfo.height
        stPixelConvertParam.ePixelFormat = frame.frameInfo.pixelFormat
        stPixelConvertParam.pSrcData = frame.pData
        stPixelConvertParam.nSrcDataLen = frame.frameInfo.size
        stPixelConvertParam.nPaddingX = frame.frameInfo.paddingX
        stPixelConvertParam.nPaddingY = frame.frameInfo.paddingY
        stPixelConvertParam.eBayerDemosaic = IMV_EBayerDemosaic.demosaicNearestNeighbor
        stPixelConvertParam.eDstPixelFormat = frame.frameInfo.pixelFormat
        stPixelConvertParam.pDstBuf = pDstBuf
        stPixelConvertParam.nDstBufSize = nDstBufSize

        # Libera el frame
        nRet = cam.IMV_ReleaseFrame(frame)
        if IMV_OK != nRet:
            print("Release frame failed! ErrorCode:", nRet)
            continue

        # Convierte el frame a formato OpenCV (cvImage)
        if stPixelConvertParam.ePixelFormat == IMV_EPixelType.gvspPixelMono8:
            imageBuff = stPixelConvertParam.pSrcData
            userBuff = c_buffer(b'\0', stPixelConvertParam.nDstBufSize)

            memmove(userBuff, imageBuff, stPixelConvertParam.nDstBufSize)
            grayByteArray = bytearray(userBuff)

            cvImage = np.array(grayByteArray).reshape(stPixelConvertParam.nHeight, stPixelConvertParam.nWidth)

        else:
            # Convierte a formato BGR24
            stPixelConvertParam.eDstPixelFormat = IMV_EPixelType.gvspPixelBGR8
            nRet = cam.IMV_PixelConvert(stPixelConvertParam)
            if IMV_OK != nRet:
                print("Image convert failed! ErrorCode:", nRet)
                continue

            rgbBuff = c_buffer(b'\0', stPixelConvertParam.nDstBufSize)
            memmove(rgbBuff, stPixelConvertParam.pDstBuf, stPixelConvertParam.nDstBufSize)
            colorByteArray = bytearray(rgbBuff)

            cvImage = np.array(colorByteArray).reshape(stPixelConvertParam.nHeight, stPixelConvertParam.nWidth, 3)

        alto, ancho = cvImage.shape[:2]
        parte_izquierda = cvImage[:, :ancho_motion]
        parte_derecha = cvImage[:, ancho - ancho_motion:]
        imagen_reducida = cv2.hconcat([parte_izquierda, parte_derecha])
        # movimiento solo en la parte izquierda
        # imagen_reducida = parte_izquierda
        # Comparar el frame actual con el frame anterior
        if prev_frame is not None and motion == False and configuration == False:

            diff = cv2.absdiff(imagen_reducida, prev_frame)
            # Convertir la diferencia a escala de grises
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            # Aplicar umbral para obtener una máscara binaria
            _, mask = cv2.threshold(gray_diff, min_threshold, 255, cv2.THRESH_BINARY)
            # Contar el número de píxeles blancos (que han cambiado)
            num_changed_pixels = np.sum(mask > 0)
            print(num_changed_pixels)
            # Si el número de píxeles que han cambiado es mayor que un umbral, consideramos que hay movimiento
            if num_changed_pixels > min_pix_changed:  # Puedes ajustar este valor según tus necesidades
                motion = True
                print("motion_detected")

        if motion == True:
            frame_list.append(cvImage)

        if configuration:
            cv2.imshow("cvImage", cvImage)
            cv2.waitKey(1)

        # Actualizar el frame anterior con el frame actual
        prev_frame = imagen_reducida.copy()

def save_frames_as_video(frames, num_changed_pixels, Exposure, FPS_real):
    current_time = datetime.datetime.now()
    folder_name = "videos"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    filename = os.path.join(folder_name, current_time.strftime("%Y-%m-%d_%H-%M-%S_") + str(num_changed_pixels) + "_px_" + str(Exposure) + "_Expo_"+ str(FPS_real) + "_FPS"+ ".avi")
    height, width, _ = frames[0].shape
    video_writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))
    for frame in frames:
        video_writer.write(frame)
    video_writer.release()
    print("Video saved successfully!")


def displayDeviceInfo(deviceInfoList):
    print("Idx  Type   Vendor              Model           S/N                 DeviceUserID    IP Address")
    print("------------------------------------------------------------------------------------------------")
    for i in range(0, deviceInfoList.nDevNum):
        pDeviceInfo = deviceInfoList.pDevInfo[i]
        strType = ""
        strVendorName = ""
        strModeName = ""
        strSerialNumber = ""
        strCameraname = ""
        strIpAdress = ""
        for str in pDeviceInfo.vendorName:
            strVendorName = strVendorName + chr(str)
        for str in pDeviceInfo.modelName:
            strModeName = strModeName + chr(str)
        for str in pDeviceInfo.serialNumber:
            strSerialNumber = strSerialNumber + chr(str)
        for str in pDeviceInfo.cameraName:
            strCameraname = strCameraname + chr(str)
        for str in pDeviceInfo.DeviceSpecificInfo.gigeDeviceInfo.ipAddress:
            strIpAdress = strIpAdress + chr(str)
        if pDeviceInfo.nCameraType == typeGigeCamera:
            strType = "Gige"
        elif pDeviceInfo.nCameraType == typeU3vCamera:
            strType = "U3V"
        print("[%d]  %s   %s    %s      %s     %s           %s" % (
            i + 1, strType, strVendorName, strModeName, strSerialNumber, strCameraname, strIpAdress))


if __name__ == "__main__":
    deviceList = IMV_DeviceList()
    interfaceType = IMV_EInterfaceType.interfaceTypeAll

    # Enumera los dispositivos
    nRet = MvCamera.IMV_EnumDevices(deviceList, interfaceType)
    if IMV_OK != nRet:
        print("Enumeration devices failed! ErrorCode", nRet)
        sys.exit()
    if deviceList.nDevNum == 0:
        print("find no device!")
        sys.exit()

    print("deviceList size is", deviceList.nDevNum)

    displayDeviceInfo(deviceList)

    #nConnectionNum = input("Please input the camera index: ")
    nConnectionNum = 1
    if int(nConnectionNum) > deviceList.nDevNum:
        print("input error!")
        sys.exit()

    cam = MvCamera()
    # Crea el manejador del dispositivo
    nRet = cam.IMV_CreateHandle(IMV_ECreateHandleMode.modeByIndex, byref(c_void_p(int(nConnectionNum) - 1)))
    if IMV_OK != nRet:
        print("Create devHandle failed! ErrorCode", nRet)
        sys.exit()

    # Abre la cámara
    nRet = cam.IMV_Open()
    if IMV_OK != nRet:
        print("Open devHandle failed! ErrorCode", nRet)
        sys.exit()
    while configuration:

        nRet = cam.IMV_SetBoolFeatureValue("AcquisitionFrameRateEnable", "True")
        if IMV_OK != nRet:
            print("Set AcquisitionFrameRateEnable failed! ErrorCode[%d]" % nRet)
            sys.exit()

        Height = input("Introduce el alto del frame (64-1080, incrementos de 4):  ")
        Height = int(Height)
        nRet = cam.IMV_SetIntFeatureValue("Height", Height)
        if IMV_OK != nRet:
            print("Set Height value failed! ErrorCode[%d]" % nRet)
            sys.exit()

        MaxOffset = 1080 - Height
        OffsetY = input("Introduce el Offset Y (max {}): ".format(MaxOffset))
        OffsetY = int(OffsetY)
        nRet = cam.IMV_SetIntFeatureValue("OffsetY", OffsetY)
        if IMV_OK != nRet:
            print("Set OffsetY value failed! ErrorCode[%d]" % nRet)
            sys.exit()

        Brightness = input("Introduce el brillo objetivo (0-100):  ")
        Brightness = int(Brightness)
        nRet = cam.IMV_SetIntFeatureValue("Brightness", Brightness)
        if IMV_OK != nRet:
            print("Set Brightness value failed! ErrorCode[%d]" % nRet)
            sys.exit()

        nRet = cam.IMV_SetEnumFeatureSymbol("ExposureAuto", "Continuous")
        if IMV_OK != nRet:
            print("Set ExposureAuto failed! ErrorCode[%d]" % nRet)
            sys.exit()

        ExposureTime = c_double(0.0)
        nRet = cam.IMV_GetDoubleFeatureValue("ExposureTime", ExposureTime)
        print("ExposureTime: ",ExposureTime.value)

        nRet = cam.IMV_SetEnumFeatureSymbol("BalanceWhiteAuto", "Continuous")
        if IMV_OK != nRet:
            print("Set BalanceWhiteAuto failed! ErrorCode[%d]" % nRet)
            sys.exit()

        FrameRate = input("Introduce el frame rate (1-2000):  ")
        FrameRate = float(FrameRate)
        nRet = cam.IMV_SetDoubleFeatureValue("AcquisitionFrameRate", FrameRate)
        if IMV_OK != nRet:
            print("Set FrameRate value failed! ErrorCode[%d]" % nRet)
            sys.exit()

        nRet = cam.IMV_StartGrabbing()
        if IMV_OK != nRet:
            print("Start grabbing failed! ErrorCode", nRet)
            sys.exit()

        try:
            g_isRunThread = True
            # Inicia el hilo para la adquisición de imágenes
            hGrabbingThread = threading.Thread(target=frameGrabbingProc, args=(cam,))
            hGrabbingThread.start()

            # Inicia el hilo para el procesamiento de frames
            hProcessingThread = threading.Thread(target=processAndStoreFrame)
            hProcessingThread.start()
        except Exception as e:
            print("Error:", e)

        FPS = c_double(0.0)
        nRet = cam.IMV_GetDoubleFeatureValue("AcquisitionFrameRate", FPS)
        print("FPS: ", FPS.value)


        nRet = cam.IMV_GetDoubleFeatureValue("ExposureTime", ExposureTime)
        print("ExposureTime: ", ExposureTime.value)

        time.sleep(0.1)

        configurado = int(input("fin de configuración? (1 = sí): "))

        if configurado == 1:
            configuration = False

        else:
            pass

        nRet = cam.IMV_StopGrabbing()
        if IMV_OK != nRet:
            print("Stop grabbing failed! ErrorCode", nRet)
            sys.exit()

    while True:
        nRet = cam.IMV_StartGrabbing()
        if IMV_OK != nRet:
            print("Start grabbing failed! ErrorCode", nRet)
            sys.exit()

        ExposureTime = c_double(0.0)
        nRet = cam.IMV_GetDoubleFeatureValue("ExposureTime", ExposureTime)
        print("ExposureTime: ",ExposureTime.value)

        try:
            g_isRunThread = True
            # Inicia el hilo para la adquisición de imágenes
            hGrabbingThread = threading.Thread(target=frameGrabbingProc, args=(cam,))
            hGrabbingThread.start()

            # Inicia el hilo para el procesamiento de frames
            hProcessingThread = threading.Thread(target=processAndStoreFrame)
            hProcessingThread.start()
        except Exception as e:
            print("Error:", e)

        while len(frame_list) == 0:
            time.sleep(0.1)

        time.sleep(t)

        # Detiene el hilo de captura
        g_isRunThread = False
        hGrabbingThread.join()
        # Espera a que el hilo de procesamiento termine
        hProcessingThread.join()

        nRet = cam.IMV_StopGrabbing()
        print("stopgrabbing")
        if IMV_OK != nRet:
            print("Stop grabbing failed! ErrorCode", nRet)
            sys.exit()
        FPS_real = int(len(frame_list) / t)
        print("FPS: ", len(frame_list) / t)
        save_frames_as_video(frame_list, num_changed_pixels, ExposureTime.value, FPS_real)
        motion = False
        frame_list = []
        prev_frame = None

    # 关闭相机
    nRet = cam.IMV_Close()
    if IMV_OK != nRet:
        print("Close camera failed! ErrorCode", nRet)
        sys.exit()

    # 销毁句柄
    if (cam.handle):
        nRet = cam.IMV_DestroyHandle()

    print("---Demo end---")