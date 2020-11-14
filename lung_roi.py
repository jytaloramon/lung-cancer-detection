####### import #######
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage.segmentation import chan_vese
from functools import cmp_to_key


def main():

    ####### Confs #######
    path_base = './'
    path_data = './data'
    path_img = './images'
    file_dataset = 'dataset0_images.npy'

    dist_check = 10
    intensity_pixel = 10

    ####### Carregando o dataset #######
    os.chdir(path_base)
    dataset = np.load(f'{path_data}/{file_dataset}', allow_pickle=True)

    # Carregando a imagem para o processamento.
    # As imagem devem está em preto e branco no intervalo 0-255. Faça as transformações necessárias.
    idx = 10
    img = dataset[idx]
    img_copy = prepare_image(img, intensity_pixel, dist_check)

    mask = run_segmentation(img_copy)
    # Extraindo a região de interesse com aplicação da máscara sobre a imagem original
    img_end = np.multiply(mask, img)

    # Visualização
    fig = plt.figure(figsize=(10, 10))
    fig.add_subplot(221).imshow(img, 'gray', vmin=0, vmax=1)
    fig.add_subplot(222).imshow(img_copy, 'gray', vmin=0, vmax=255)
    fig.add_subplot(223).imshow(mask, 'gray', vmin=0, vmax=1)
    fig.add_subplot(224).imshow(img_end, 'gray', vmin=0, vmax=1)
    plt.show()

    # Salvando a imagem com o processo
    # fig.savefig(f'{path_img}/fig_1.png')


def prepare_image(img, intensity_pixel, dist_check):
    """ Efetua tranformação intensidade dos pixeis da imagem e aplicação filtro de suavização

    Parameters
    ---------
    img: matriz de píxel com valores no intervalo 0-1.\n
    intensity_pixel: valor máximo, todos os pixeis abaixo desse valor será alterado.\n   
    dist_check: distância do píxel que será checado,
        o valor contido nessa localidade será a base da normalização. img[center][dist_check]. 

    Returns:
    ---------
    matriz: uma cópia da matriz original com as transformações, reescala para intervalo 0-255.
    """
    filter_sigma_x = 5
    filter_kernel = (17, 17)
    img_border_padding = 5
    new_img = np.copy(np.uint8(img * 255))

    new_img[new_img < intensity_pixel] = new_img[dist_check][new_img.shape[0] // 2]
    new_img[:img_border_padding] = 0
    new_img = cv2.GaussianBlur(new_img, filter_kernel, filter_sigma_x)

    return new_img


def extract_coord_box(idx, list_pointer):
    """ Retorna os pontos extremos de um contorno, formando uma caixa.

    Parameters
    ---------
    idx: índex do contorno.\n
    list_pointer: lista de pontos que formam o contorno.\n

    Returns:
    ---------
    tuple: (idx, p_top, p_right, p_down, p_left) - sentido horário.
    """

    p_left = p_top = 512
    p_right = p_down = 0

    for pointer in list_pointer:
        if pointer[0][1] < p_top:
            p_top = pointer[0][1]
        elif pointer[0][1] > p_down:
            p_down = pointer[0][1]

        if pointer[0][0] < p_left:
            p_left = pointer[0][0]
        elif pointer[0][0] > p_right:
            p_right = pointer[0][0]

    return (idx, p_top, p_right, p_down, p_left)


def find_color(mask, coord, color_check=0, margin=16):
    """ Busca por uma determinada cor nas proximidades de uma coordenada

    Parameters
    ---------
    mask: matriz de pixel base da busca.\n
    coord: coordenadas extremas da caixa - (top, right, down, left).\n
    color_check: cor buscada. Default: 0 - preto.\n
    margin: distância da busca. Default: 16.

    Returns:
    ---------
    boolean: True se pelo menos houve dois lado satisfeito, False caso não.
    """

    def calc_avg(x_1, x_2): return (x_1 + x_2) // 2

    qt_to_consensu = 2
    x_center = calc_avg(coord[1], coord[3])
    y_center = calc_avg(coord[0], coord[2])
    coods_to_check = [(coord[0] - margin, x_center),
                      (coord[2] + margin, x_center),
                      (y_center, coord[1] + margin),
                      (y_center, coord[3] - margin)
                      ]

    i = count = 0
    while i < len(coods_to_check) and count < qt_to_consensu:
        if mask[coods_to_check[i][0]][coods_to_check[i][1]] == color_check:
            count += 1
        i += 1

    return count >= qt_to_consensu


def filter_box(mask, boxes_contours, min_dist_border=16):
    """ Remove os boxes que não sastifazem as condições\n  
    Condições: devem está a uma distância mínima das bordas
        e devem ter a cor branca internamente.

    Parameters
    ---------
    mask: matriz de pixel base da busca.\n
    boxes_contours: lista coordenadas extremas da boxes - (idx, top, right, down, left).\n 
    min_dist_border: mínima distância para as bordas. Default: 16.

    Returns:
    ---------
    list: boxes que sastifazem as condições 
    """

    dimension_mask = 512

    box_list = []
    for box_coord in boxes_contours:
        coord = (box_coord[1], box_coord[2], box_coord[3], box_coord[4])
        if (not (box_coord[1] < min_dist_border or box_coord[4] < min_dist_border
                 or box_coord[2] + min_dist_border >= dimension_mask or box_coord[3] + min_dist_border >= dimension_mask)
                and find_color(mask, coord)):

            box_list.append(box_coord)

    return box_list


def remove_duplicate_box(boxes):
    """ Remove boxes que têm suas extremidade repetidas - duplicados

    Parameters
    ---------
    boxes: lista de boxes.\n

    Returns:
    ---------
    list: lista de índexes sem repetição de coordenadas.
    """

    coord = {}
    new_list_boxes = []

    for box in boxes:
        key_box = str(box[1]) + str(box[1]) + str(box[2]) + str(box[3])
        if coord.get(key_box) is None:
            coord[key_box] = True
            new_list_boxes.append(box[0])

    return new_list_boxes


def run_segmentation(img, threshold=10):
    """ Executa a segmentação da imagem

    Parameters
    ---------
    img: matriz de píxel com valores no intervalo 0-255.\n
    threshold: valor limite para a extração de bordas.

    Returns
    ---------
    matriz: máscara da imagem.
    """

    # Criando as matrizes de segmentação
    img_segmentation = np.zeros(img.shape, dtype=np.uint8)
    mask = np.zeros(img.shape, dtype=np.uint8)

    # Segmentação da imagem
    # Convertendo a imagem segmentada - máscara - em valores ou 0 ou 1.
    cv = chan_vese(img_as_float(img), mu=0.25, lambda1=1, lambda2=1, tol=1e-3, max_iter=200,
                   dt=0.5, init_level_set="checkerboard", extended_output=True)
    img_segmentation[cv[0] == True], img_segmentation[cv[0] == False] = 0, 255

    # Extração de contorno da segmentação.
    canny_output = cv2.Canny(img_segmentation, threshold, threshold * 2)
    contours, hierarchy = cv2.findContours(
        canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Criando uma caixa(virtual) para o contorno, usando os pontos mais extremos.
    # Retorna um lista de tuplas no formato: (idx_contorno_na_lista, p_topo, p_direita, p_baixo, p_esquerdo)
    boxes_all_contours = map(extract_coord_box, range(len(contours)), contours)

    # Filtrando as caixas e remoção de pontos duplicados
    # Retorna apenas os índexes dos contornos das regiões que satisfazem as condições
    idx_contours = remove_duplicate_box(
        filter_box(img_segmentation, boxes_all_contours))

    # Aplicando Convex Hull nos contornos
    hull_list = list(
        map(lambda idx: cv2.convexHull(contours[idx]), idx_contours))

    # Pintando as regiões para a criação da máscara final
    cv2.drawContours(mask, hull_list, -1, 1, thickness=-1)

    # Dilatando a máscara
    mask = cv2.dilate(mask, kernel=cv2.getStructuringElement(
        cv2.MORPH_DILATE, ksize=(5, 5)), iterations=6)

    return mask


if __name__ == "__main__":
    main()
