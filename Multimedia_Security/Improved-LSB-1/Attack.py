import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import metrics

def PSNR(img1, img2, data_range=255):
    PSNR = metrics.peak_signal_noise_ratio(img1, img2, data_range=data_range)
    return PSNR

def SSIM(img1, img2):
    SSIM = metrics.structural_similarity(img1, img2, full=True, win_size=7)
    return SSIM[0]

def BER(info, img1, img2):
    # 將圖片轉換為一维數組
    img1_bits = img1.flatten()
    img2_bits = img2.flatten()

    # 將0,1數值轉為0,255
    img1_bits *= 255

    # 如果是MeanFilter或MedianFilter取出浮水印圖會黑白相反
    if info == 'MeanFilter' or info == 'MedianFilter':
        img2_bits = 255 - img2_bits

    # 計算錯誤bit數量
    error_bits = np.sum(img1_bits != img2_bits)

    # 計算BER
    total_bits = len(img1_bits)
    BER = error_bits / total_bits
    return BER

# 添加高斯噪聲
def gaussian_noise(image, mean=0, sigma=25):
    row, col = image.shape
    gauss = np.random.normal(mean, sigma, (row, col))
    noisy = image + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

# 添加椒鹽噪聲
def salt_and_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02):
    row, col = image.shape
    noisy = image.copy()
    salt = np.random.rand(row, col) < salt_prob
    pepper = np.random.rand(row, col) < pepper_prob
    noisy[salt] = 255
    noisy[pepper] = 0
    return noisy

# 添加均值濾波
def mean_filter(image, kernel_size=3):
    return cv2.blur(image, (kernel_size, kernel_size))

# 添加中值濾波
def median_filter(image, kernel_size=3):
    return cv2.medianBlur(image, kernel_size)

# 添加高通濾波
def high_pass_filter(image):
    kernel = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)

# 旋轉圖像
def rotate_image(image, angle=45):
    row, col = image.shape
    rotation_matrix = cv2.getRotationMatrix2D((col / 2, row / 2), angle, 1)
    return cv2.warpAffine(image, rotation_matrix, (col, row))

def attack_and_evaluate(attack_type, original_image):
    attacked_image = globals()[attack_type](original_image)
    psnr = PSNR(original_image, attacked_image)
    ssim = SSIM(original_image, attacked_image)
    ber = BER(attack_type, original_image, attacked_image)
    return attacked_image, psnr, ssim, ber

# 主程序
def main():
    # 載入原始圖像(修改過後)
    original_image = cv2.imread("Improved-LSB-Steganography/images/out1.png", 0)

    attacks = ['gaussian_noise', 'salt_and_pepper_noise', 'mean_filter',
               'median_filter', 'high_pass_filter', 'rotate_image']


    # 調整行和列的數量
    num_rows = 2
    num_cols = 3

    plt.figure(figsize=(10, 6))
    plt.subplot(num_rows, num_cols, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Original Image')

    for i, attack_type in enumerate(attacks, start=1):
        attacked_image, psnr, ssim, ber = attack_and_evaluate(attack_type, original_image)
        
        # 調整行和列的數量
        row = (i - 1) // num_cols + 1
        col = (i - 1) % num_cols + 1
        
        plt.subplot(num_rows, num_cols, i)
        plt.imshow(attacked_image, cmap='gray')
        plt.title(f'{attack_type}\nPSNR: {psnr:.2f}, SSIM: {ssim:.2f}, BER: {ber:.4f}')
    
    plt.tight_layout()  # 調整子圖之間的間距
    plt.savefig(os.path.join('Improved-LSB-Steganography/result', 'all_results.png'))
    plt.show()

    # Write results to a text file
    with open('Improved-LSB-Steganography/result/attack_results.txt', 'w') as f:
        f.write('Attack Results for Improved-LSB-Steganography\n\n')
        for attack_type in attacks:
            attacked_image, psnr, ssim, ber = attack_and_evaluate(attack_type, original_image)
            f.write(f'Attack Type: {attack_type}\n')
            f.write(f'PSNR: {psnr:.2f}, SSIM: {ssim:.2f}, BER: {ber:.4f}\n\n')

if __name__ == "__main__":
    main()
