from __future__ import print_function
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import frangi
import cv2 as cv

# file1_list = ['./eyes/healthy/healthy/01_h.jpg']
file1_list = ['./eyes/healthy/healthy/01_h.jpg',
              # './eyes/diabetic_retinopathy/diabetic_retinopathy/02_dr.JPG']
              './eyes/diabetic_retinopathy/diabetic_retinopathy/02_dr.JPG',
              # './eyes/diabetic_retinopathy/diabetic_retinopathy/03_dr.JPG']
              './eyes/diabetic_retinopathy/diabetic_retinopathy/03_dr.JPG',
              './eyes/glaucoma/glaucoma/04_g.jpg',
              './eyes/glaucoma/glaucoma/05_g.jpg']
file2_list = ['./eyes/healthy/healthy_manualsegm/01_h.tif',
              './eyes/diabetic_retinopathy/diabetic_retinopathy_manualsegm/02_dr.tif',
              './eyes/diabetic_retinopathy/diabetic_retinopathy_manualsegm/03_dr.tif',
              './eyes/glaucoma/glaucoma_manualsegm/04_g.tif',
              './eyes/glaucoma/glaucoma_manualsegm/05_g.tif']
file3_list = ['./eyes/healthy/healthy_fovmask/01_h_mask.tif',
              './eyes/diabetic_retinopathy/diabetic_retinopathy_fovmask/02_dr_mask.tif',
              './eyes/diabetic_retinopathy/diabetic_retinopathy_fovmask/03_dr_mask.tif',
              './eyes/glaucoma/glaucoma_fovmask/04_g_mask.tif',
              './eyes/glaucoma/glaucoma_fovmask/05_g_mask.tif']


def get_files(file1, file2, file3):
    print('Get files')
    original_image = io.imread(file1)
    hand_drawn = cv.imread(file2, 0)
    fov_image = cv.imread(file3, 0)
    return original_image, hand_drawn, fov_image


def grey_filter(img):
    print('Grey filter')
    r, g, b = cv.split(img)
    equalized_g = cv.equalizeHist(g)
    return cv.merge((equalized_g, equalized_g, equalized_g))


def frangi_filter(img):
    print('Frangi filter')
    blur = cv.GaussianBlur(img, (15, 15), 23)
    weight = cv.addWeighted(img, 1.5, blur, -0.5, 0, blur)
    return frangi(weight)


def result_frangi_fov(frangi_img, fov_img):
    print('Result frangi fov')
    result_img = frangi_img * fov_img
    max_value = 0
    min_value = 1
    for i in range(len(result_img)):
        for j in range(len(result_img[0])):
            if result_img[i][j] > max_value:
                max_value = result_img[i][j]
            elif result_img[i][j] < min_value:
                min_value = result_img[i][j]

    for i in range(len(result_img)):
        for j in range(len(result_img[0])):
            result_img[i][j] = (result_img[i][j] - min_value) / (max_value - min_value)
            result_img[i][j] = result_img[i][j] * 255

    kernel = np.ones((5, 5), np.float32) / 25
    result_img = cv.filter2D(result_img, -1, kernel)
    ret, thresh = cv.threshold(result_img, 200, 255, cv.THRESH_BINARY)
    kernel = np.ones((25, 25), np.uint8)
    erode = cv.erode(thresh, kernel, iterations=10)
    result_img = cv.bitwise_or(result_img, erode)
    return result_img


def get_processed_image(img):
    print('Get processed image')
    for i in range(len(img)):
        for j in range(len(img[0])):
            if img[i][j] > 5:
                img[i][j] = 255
            else:
                img[i][j] = 0
    return img / 255


def create_mistake_matrix(hand_drawn_img, mask_img, processed_image):
    print('Create mistake matrix')
    mistake_matrix = np.zeros((len(hand_drawn_img), len(hand_drawn_img[0])), dtype=bool)

    compliance = 0
    non_compliance = 0
    for i in range(len(hand_drawn_img)):
        for j in range(len(hand_drawn_img[0])):
            if mask_img[i][j] == 1 and hand_drawn_img[i][j] != processed_image[i][j]:
                mistake_matrix[i][j] = 1
            elif mask_img[i][j] == 1 and hand_drawn_img[i][j] == processed_image[i][j]:
                mistake_matrix[i][j] = 0
                if hand_drawn_img[i][j] == 1:
                    compliance += 1
                else:
                    non_compliance += 1
    return mistake_matrix, compliance, non_compliance


def process(file1, file2, file3):
    original_image, hand_drawn, fov_image = get_files(file1, file2, file3)

    grey_image = grey_filter(original_image)

    color_image = cv.cvtColor(grey_image, cv.COLOR_RGB2GRAY)

    frangi_image = frangi_filter(color_image)

    result_image = result_frangi_fov(frangi_image, fov_image)

    processed_image = get_processed_image(result_image)

    hand_drawn = hand_drawn.astype(bool)

    mask = io.imread(file3, as_gray=True)
    mistake_matrix, compliance, non_compliance = create_mistake_matrix(hand_drawn, mask, processed_image)

    sensitivity = compliance / hand_drawn.sum()
    specificity = non_compliance / (hand_drawn.size - (mask.size - mask.sum()) - hand_drawn.sum())
    accuracy = (compliance + non_compliance) / (hand_drawn.size - (mask.size - mask.sum()))
    return original_image, hand_drawn, processed_image, mistake_matrix, sensitivity, specificity, accuracy


if __name__ == '__main__':
    fig = plt.figure(figsize=(20, 20))

    counter = 0
    for i in range(len(file1_list)):
        print('Files ' + file1_list[i] + ', ' + file2_list[i] + ', ' + file3_list[i])
        image, hand_drawn, processed_image, mistake_matrix, sensitivity, specificity, accuracy = process(
            file1=file1_list[i], file2=file2_list[i], file3=file3_list[i])

        counter += 1
        fig.add_subplot(len(file1_list)+1, 4, counter)
        plt.title('Original')
        textstr = 'Sensitivity: ' + str(round(sensitivity * 100, 2)) + '%\n' + \
                  'Specificity: ' + str(round(specificity * 100, 2)) + '%\n' + \
                  'Accuracy: ' + str(round(accuracy * 100, 2)) + '%\n'
        plt.text(1, .4, textstr, bbox=dict(boxstyle="square", facecolor="white"))
        plt.imshow(image)

        counter += 1
        fig.add_subplot(len(file1_list)+1, 4, counter)
        plt.title('Hand-drawn')
        plt.imshow(hand_drawn)

        counter += 1
        fig.add_subplot(len(file1_list)+1, 4, counter)
        plt.title('Code processed image')
        plt.imshow(processed_image)

        counter += 1
        fig.add_subplot(len(file1_list)+1, 4, counter)
        plt.title('Mistake matrix')
        plt.imshow(mistake_matrix, cmap='gray')

    fig.savefig('output.jpg', bbox_inches='tight')
