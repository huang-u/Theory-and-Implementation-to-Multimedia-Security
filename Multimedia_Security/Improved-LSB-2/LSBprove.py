import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import metrics
import os
def PSNR(img1, img2, data_range=255):
    PSNR = metrics.peak_signal_noise_ratio(img1, img2, data_range=data_range)
    return PSNR

def SSIM(img1, img2):
    SSIM = metrics.structural_similarity(img1, img2, full=True, win_size=7)
    return SSIM[0]

def BER(info, img1, img2):
    img1_bits = img1.flatten() * 255
    img2_bits = img2.flatten()
    
    if info == 'MeanFilter' or info == 'MedianFilter':
        img2_bits = 255 - img2_bits

    error_bits = np.sum(img1_bits != img2_bits)
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

def median_filter(image, kernel_size=3):
    # Convert the image to uint8 if it's not already
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    # Apply median filter
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

def lsb_matching(cover_image, message_image, M, N, m, n):
    for i in range(m):
        for j in range(0, min(n * 2, N - 1), 2):  # Adjusted loop bounds
            if (j == n * 2 - 2 or j == n * 2 - 3) and i < m - 1:
                i += 1

            if j + 1 < N and i < M:
                if message_image[i, j] == cover_image[i, j] % 2:
                    if j + 1 < N and i < M and message_image[i, j + 1] != (cover_image[i, j] // 2 + cover_image[i, j + 1]) % 2:
                        if cover_image[i, j + 1] % 2 == 1:
                            cover_image[i, j + 1] -= 1
                        elif cover_image[i, j + 1] % 2 == 0:
                            cover_image[i, j + 1] += 1
                    else:
                        cover_image[i, j + 1] = cover_image[i, j + 1]
                    cover_image[i, j] = cover_image[i, j]
                elif j + 1 < N and i < M and message_image[i, j] != cover_image[i, j] % 2:
                    if j + 1 < N and i < M and message_image[i, j + 1] == (cover_image[i, j] // 2 - 1 + cover_image[i, j + 1]) % 2:
                        cover_image[i, j] -= 1
                    else:
                        cover_image[i, j] += 1
                    cover_image[i, j + 1] = cover_image[i, j + 1]

    return cover_image
def extract_lsb_matching(hidden_image, M, N, m, n):
    extracted_message = np.zeros((m, n), dtype=np.uint8)

    for i in range(m):
        for j in range(0, min(n * 2, N - 1), 2):
            if (j == n * 2 - 2 or j == n * 2 - 3) and i < m - 1:
                i += 1

            if j + 1 < N and i < M:
                # Extract the least significant bit
                extracted_message[i, j] = hidden_image[i, j] % 2
                extracted_message[i, j + 1] = hidden_image[i, j + 1] % 2

    # Map back to [0, 255]
    extracted_message *= 255

    return extracted_message
def apply_attack(image, attack_type):
    if attack_type == 'gaussian_noise':
        # Add Gaussian noise
        return gaussian_noise(image)
    elif attack_type == 'salt_and_pepper_noise':
        # Add salt and pepper noise
        return salt_and_pepper_noise(image)
    elif attack_type == 'mean_filter':
        # Apply mean filter
        return mean_filter(image)
    elif attack_type == 'median_filter':
        # Apply median filter
        return median_filter(image)
    elif attack_type == 'high_pass_filter':
        # Apply high pass filter
        return high_pass_filter(image)
    elif attack_type == 'rotate_image':
        # Rotate the image
        return rotate_image(image)
    else:
        print(f"Error: Unknown attack type '{attack_type}'")
        return image
def extract_watermark(attacked_image, watermark_shape):
    M, N = attacked_image.shape
    m, n = watermark_shape

    # 调用提取浮水印的方法
    extracted_watermark = extract_lsb_matching(attacked_image, M, N, m, n)
    return extracted_watermark

def attack_and_evaluate(attack_type, original_image, original_watermark):
    # Apply the selected attack to the original image
    attacked_image = apply_attack(original_image, attack_type)
    
    # Extract the watermark from the attacked image
    attacked_watermark = extract_watermark(attacked_image, original_watermark.shape)
    
    # Evaluate the attack
    psnr = PSNR(original_image, attacked_image)
    ssim = SSIM(original_watermark, attacked_watermark)
    ber = BER(attack_type, original_watermark, attacked_watermark)

    # Save the attacked image
    attacked_image_filename = f'{attack_type}.png'
    attacked_image_path = os.path.join('Improved-LSB-2/image', attacked_image_filename)
    cv2.imwrite(attacked_image_path, attacked_image)
    
    return attacked_image, attacked_watermark, psnr, ssim, ber
# Load images with error handling
cover_image = cv2.imread('Improved-LSB-2/image/lena.bmp', cv2.IMREAD_GRAYSCALE)
# ... (same as before)

message_image = cv2.imread('Improved-LSB-2/image/airplane.bmp', cv2.IMREAD_GRAYSCALE)
# ... (same as before)

_, message_image = cv2.threshold(message_image, 128, 255, cv2.THRESH_BINARY)

# Get image dimensions
M, N = cover_image.shape
m, n = message_image.shape

# Apply LSB matching algorithm
hide_image = lsb_matching(cover_image.astype(float), message_image.astype(float), M, N, m, n)

# List of attack types to apply
attack_types = ['gaussian_noise', 'salt_and_pepper_noise', 'mean_filter', 'median_filter', 'high_pass_filter', 'rotate_image']

# Load images with error handling
cover_image = cv2.imread('Improved-LSB-2/image/lena.bmp', cv2.IMREAD_GRAYSCALE)
message_image = cv2.imread('Improved-LSB-2/image/airplane.bmp', cv2.IMREAD_GRAYSCALE)
_, message_image = cv2.threshold(message_image, 128, 255, cv2.THRESH_BINARY)

# Get image dimensions
M, N = cover_image.shape
m, n = message_image.shape

# Apply LSB matching algorithm
hide_image = lsb_matching(cover_image.astype(float), message_image.astype(float), M, N, m, n)

with open('Improved-LSB-2/txt/attack_results.txt', 'w') as file:
    file.write('Results for Improved-LSB-2\n\n')
    # Loop through each attack type
    for attack_type in attack_types:
        # Apply the selected attack to the hidden image
        attacked_image = apply_attack(hide_image, attack_type)

        # Save the attacked image
        attacked_image_filename = f'attacked_image_{attack_type}.bmp'
        attacked_image_path = os.path.join('Improved-LSB-2/image', attacked_image_filename)
        cv2.imwrite(attacked_image_path, attacked_image.astype(np.uint8))

        # Extract the watermark from the attacked image
        attacked_watermark = extract_watermark(attacked_image, message_image.shape)

        # Evaluate the attack
        psnr = PSNR(hide_image, attacked_image)
        ssim = SSIM(message_image, attacked_watermark)
        ber = BER(attack_type, message_image, attacked_watermark)

        # Display images (including the attacked image)
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 4, 1), plt.imshow(cover_image, cmap='gray'), plt.title('Cover Image')
        plt.subplot(1, 4, 2), plt.imshow(message_image, cmap='gray'), plt.title('Message Image')
        plt.subplot(1, 4, 3), plt.imshow(hide_image, cmap='gray'), plt.title('Hidden Image')
        plt.subplot(1, 4, 4), plt.imshow(attacked_image, cmap='gray'), plt.title(f'Attacked Image ({attack_type})')
        plt.show()

        # Extract the hidden message using LSB extraction
        extracted_message = extract_lsb_matching(attacked_image.astype(float), M, N, m, n)

        # Save the extracted message as an image
        extracted_message_filename = f'extracted_message_{attack_type}.bmp'
        attacked_image_path = os.path.join('Improved-LSB-2/image', attacked_image_filename)
        cv2.imwrite(attacked_image_path, extracted_message.astype(np.uint8))

        # Display the extracted message
        plt.figure(figsize=(5, 4))
        plt.imshow(extracted_message, cmap='gray')
        plt.title('Extracted Message')
        plt.show()

        # Calculate SSIM between original and extracted watermark
        ssim_original_vs_extracted = SSIM(message_image, extracted_message)
        print(f'Attack Type: {attack_type}')
        print(f'PSNR: {psnr:.4f}')
        print(f'SSIM between original and extracted watermark: {ssim_original_vs_extracted:.4f}')
        print(f'BER: {ber:.4f}')
        print('-' * 40)
        file.write(f'Attack Type: {attack_type}\n')
        file.write(f'PSNR: {psnr:.4f}\n')
        file.write(f'SSIM between original and extracted watermark: {ssim_original_vs_extracted:.4f}\n')
        file.write(f'BER: {ber:.4f}\n')
        file.write('-' * 40 + '\n')
