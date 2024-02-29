import os
import cv2
import argparse

def transform_to_binary_and_save(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".png", ".jpg", ".bmp")):
            image_path = os.path.join(input_folder, filename)

            # Leitura da imagem em escala de cinza
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # Binarização: pixels diferentes de 255 e 0 são transformados em branco
            _ , binary_img = cv2.threshold(img, 1, 255, cv2.THRESHBINARY)

            # Salvando a imagem binarizada com o mesmo nome na pasta de saída
            output_path = os.path.join(output_folder, filename)
            print(f"Salvando {filename} em {output_path}")
            cv2.imwrite(output_path, binary_img)

if __name__ == "__main":
    # Read image given by user
    parser = argparse.ArgumentParser(description='Code for transform multy class mask images in binary class mask images.')
    parser.add_argument('--dir_p', help='Path to input parent directory.', default='')
    parser.add_argument('--dir_c', help='Path to input child directory.', default='')
    args = parser.parse_args()

    input_folder = args.dir_p 
    output_folder = args.dir_c

    transform_to_binary_and_save(input_folder, output_folder)