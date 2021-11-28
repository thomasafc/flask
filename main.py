import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

from PIL import Image
from tensorflow import keras

train_size        = 1 # Tamanho dos dados para treinamento
max_vid_per_class = 1 # Número máximo de vídeos por classe
num_of_frames     = 12 # Número de frames que o vídeo será dividido
width             = 64 # Largura da imagem
height            = 48 # Altura da imagem
num_of_channels   = 1  # Número de canais da imagem (cinza, RGB, etc.)
num_of_classes    = 2  # Número de classes que serão utilizadas

# class_names = {
#     "Acontecer"  : 0, 
#     "Aluno"      : 1, 
#     "Amarelo"    : 2, 
#     "America"    : 3, 
#     "Aproveitar" : 4, 
#     "Bala"       : 5, 
#     "Banco"      : 6, 
#     "Banheiro"   : 7, 
#     "Barulho"    : 8, 
#     "Cinco"      : 9
# }

class_names = {
   "A"          : 0, 
   "Acontecer"  : 1, 
   "Aluno"      : 2, 
   "Amarelo"    : 3, 
   "America"    : 4, 
   "Aproveitar" : 5, 
   "B"          : 6, 
   "C"          : 7, 
   "D"          : 8, 
   "E"          : 9 
}

def convertVideo(videoName):
    dataVideo = [] # armazena os frames

    # caminho do vídeo
    vid = str('./videoUpload/' + videoName)

    # lê o vídeo
    cap = cv2.VideoCapture(vid)
    frames = [] # armazenar os frames
    count = 0 # contador para pegar cada frame

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    seconds = frame_count/fps
    miliseconds = seconds * 1000
    frame_moment = int(miliseconds/num_of_frames)


    for j in range(num_of_frames): # aqui pegamos n frames de acordo com num_of_frames
        # 1000 = 1 segundo, 500 = meio segundo...
        cap.set(cv2.CAP_PROP_POS_MSEC,(count*frame_moment)) # seta o momento do vídeo para pegar o frame
        ret, frame = cap.read() # pega de fato o frame, e se deu sucesso ou não
        if ret == True: # se deu sucesso...
            print('S=>', end="")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # converte o frame para cinza
            frame = cv2.resize(frame,(width,height),interpolation=cv2.INTER_AREA) # redimensiona o frame
            frames.append(frame) # adiciona o frame para o vetor de frames
        else:
            print('*E*=>', end="")
        count = count + 1
    dataVideo.append(frames) # adiciona todos os frames de um vídeo
            
    dataVideo = np.array(dataVideo) # transforma o vetor de frames em array numpy
    print('\n\n===== FIM =====\n\n')

    # transforma os dados dos frames armazenados no formato ideal para ser usado na rede
    dataVideo = dataVideo.reshape(train_size, num_of_frames, width, height, num_of_channels)
    print(np.shape(dataVideo))


    # Retorna vetor com dados do video
    return dataVideo


def convertImage(imageName):
    dataImage = [] # Armazena os frames
    frames = [] # armazenar os frames

    # Pegando imagem
    frame = np.array(Image.open('./imageUpload/' + imageName))
    
    # Convertendo imagem para tons de cinza
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Redimencionsando Imagem
    frame = cv2.resize(frame,(width,height),interpolation=cv2.INTER_AREA)

    # adiciona o frame para o vetor de frames
    for i in range(num_of_frames):
        frames.append(frame)
    # adiciona todos os frames de um vídeo
    dataImage.append(frames)

    # Convertendo imagem para frames para ser usado na rede
    dataImage = np.array(dataImage)

    # transforma os dados dos frames armazenados no formato ideal para ser usado na rede
    dataImage = dataImage.reshape(train_size, num_of_frames, width, height, num_of_channels)
    print(np.shape(dataImage))

    # Retorna vetor com dados da imagem
    return dataImage


def callModel(frames):
    print(tf.__version__)
    # Chamando o modelo da rede treinada
    newModel = keras.models.load_model('./model/')
    prediction = newModel.predict(np.expand_dims(frames[0], axis=0))[0]
    bestAccuracy = 0
    bestName = ''
    for predict, name in zip(prediction, class_names):
        accuracy = (100 * predict)
        if (accuracy > bestAccuracy):
            bestAccuracy = accuracy
            bestName = name
        print(
            "%.2f ==> %s"
            % (accuracy, name)
        )

    return bestName