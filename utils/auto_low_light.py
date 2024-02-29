from __future__ import print_function
from builtins import input
import cv2 as cv
import numpy as np
import argparse
import os

# Read image given by user
parser = argparse.ArgumentParser(description='Code for Changing the brightness on the images in a folder.')
parser.add_argument('--dir_p', help='Path to input parent directory.', default='')
parser.add_argument('--dir_c', help='Path to input child directory.', default='')
args = parser.parse_args()

def low_light(image):
    new_image = np.zeros(image.shape, image.dtype)
    alpha = 1.0 # Simple contrast control
    beta = -140    # Simple brightness control

    # Do the operation new_image(i,j) = alpha*image(i,j) + beta
    # Instead of these 'for' loops we could have used simply:
    # new_image = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
    # but we wanted to show you how to access the pixels :)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            for c in range(image.shape[2]):
                new_image[y,x,c] = np.clip(alpha*image[y,x,c] + beta, 0, 255)
    return new_image


def processar_imagens_na_pasta(dir_entrada, dir_saida):
    # Criar diretório de saída se não existir
    if not os.path.exists(dir_saida):
        os.makedirs(dir_saida)
    
    # Percorrer todos os arquivos na pasta de entrada
    for nome_arquivo in os.listdir(dir_entrada):
        caminho_entrada = os.path.join(dir_entrada, nome_arquivo)
        
        # Verificar se é um arquivo de imagem
        if nome_arquivo.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            # Ler a imagem
            imagem = cv.imread(caminho_entrada)
            
            # Aplicar a transformação de baixa luminosidade
            imagem_baixa_luminosidade = low_light(imagem)
            
            # Salvar a imagem transformada na pasta de saída
            caminho_saida = os.path.join(dir_saida, nome_arquivo)
            cv.imwrite(caminho_saida, imagem_baixa_luminosidade)
            print(f"Imagem processada e salva: {caminho_saida}")

# Diretórios de entrada e saída
dir_entrada = args.dir_p
dir_saida = args.dir_c

# Chamar a função para processar as imagens na pasta de entrada e salvar na pasta de saída
processar_imagens_na_pasta(dir_entrada, dir_saida)
