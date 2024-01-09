import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage import metrics as sk_metrics

# Adding the attack functions
def gaussian_noise(image, mean=0, sigma=25):
    row, col, _ = image.shape
    gauss = np.random.normal(mean, sigma, (row, col, 3))
    noisy = image + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def salt_and_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02):
    row, col, _ = image.shape
    noisy = image.copy()
    salt = np.random.rand(row, col) < salt_prob
    pepper = np.random.rand(row, col) < pepper_prob
    noisy[salt] = 255
    noisy[pepper] = 0
    return noisy

def mean_filter(image, kernel_size=3):
    return cv2.blur(image, (kernel_size, kernel_size))

def median_filter(image, kernel_size=3):
    return cv2.medianBlur(image, kernel_size)

def high_pass_filter(image):
    kernel = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)

def rotate_image(image, angle=45):
    row, col, _ = image.shape
    rotation_matrix = cv2.getRotationMatrix2D((col / 2, row / 2), angle, 1)
    return cv2.warpAffine(image, rotation_matrix, (col, row))

class LSB_Embed():
    def __init__(self):
        pass

    @staticmethod
    def get_bitPlane(img):
        """
        獲取彩色圖像的8個位平面
        :param img: 彩色圖像
        :return: 8個位平面的張量shape=(h, w, 8, c)
        """
        # w, h, c = img.shape
        h, w, c = img.shape
        bitPlane = np.zeros(shape=(h, w, 8, c))
        for c_id in range(c):
            flag = 0b00000001
            for bit_i in range(bitPlane.shape[-2]):
                bitplane = img[..., c_id] & flag  # 獲取圖像的某一位，從最後一位开始處理
                bitplane[bitplane != 0] = 1  # 阈值處理 非0即1
                bitPlane[..., bit_i, c_id] = bitplane  # 處理後的數據載入到某個位平面
                flag <<= 1  # 獲取下一個位的信息
        return bitPlane.astype(np.uint8)

    @staticmethod
    def lsb_embed(background, watermark, embed_bit=1):
        """
        在background的低三位進行嵌入浮水印，具體為將watermark的高三位信息替換掉background的低三位信息
        :param background: 背景圖像（彩色）
        :param watermark: 浮水印圖像（彩色）
        :return: 嵌入浮水印的圖像
        """
        # 1. 判断是否满足可嵌入的条件
        w_h, w_w,w_c = watermark.shape
        b_h,b_w, b_c = background.shape
        assert w_w < b_w and w_h < b_h, "請保證watermark尺寸小於background尺寸\r\n當前尺寸watermark:{}, background:{}".format(
            watermark.shape, background.shape)

        # 2. 獲取位平面
        bitPlane_background = lsb.get_bitPlane(
            background)  # 獲取的平面顺序是從低位到高位的 [（0 1 2 3 4 5 6 7）,（0 1 2 3 4 5 6 7）,（0 1 2 3 4 5 6 7）]
        bitPlane_watermark = lsb.get_bitPlane(watermark)

        # 3. 在位平面嵌入信息
        for c_i in range(b_c):
            for i in range(embed_bit):
                # 信息主要集中在高位，此處將watermark的高三位信息 放置在 background低三位信息中
                bitPlane_background[0:w_h, 0:w_w, i, c_i] = bitPlane_watermark[0:w_h, 0:w_w, (8 - embed_bit) + i, c_i]

        # 4. 得到watermark_img 浮水印嵌入圖像
        synthesis = np.zeros_like(background)
        for c_i in range(b_c):
            for i in range(8):
                synthesis[..., c_i] += bitPlane_background[..., i, c_i] * np.power(2, i)
        return synthesis.astype(np.uint8)

    @staticmethod
    def lsb_extract(synthesis, embed_bit=3):
        bitPlane_synthesis = lsb.get_bitPlane(synthesis)
        extract_watermark = np.zeros_like(synthesis)
        extract_background = np.zeros_like(synthesis)
        for c_i in range(3):
            for i in range(8):
                if i < embed_bit:
                    extract_watermark[..., c_i] += bitPlane_synthesis[..., i, c_i] * np.power(2, (8 - embed_bit) + i)
                else:
                    extract_background[..., c_i] += bitPlane_synthesis[..., i, c_i] * np.power(2, i)
        return extract_watermark.astype(np.uint8), extract_background.astype(np.uint8)

    @staticmethod
    def resize_images(img1, img2):
        """
        Resize img2 to the size of img1 while maintaining the aspect ratio.
        """
        h1, w1, _ = img1.shape
        h2, w2, _ = img2.shape

        # Resize img2 to have the same height as img1
        img2_resized = cv2.resize(img2, (w1, int(h2 * w1 / w2)))

        # Pad the resized image with zeros to match the height of img1
        pad_height = h1 - img2_resized.shape[0]
        img2_resized = np.pad(img2_resized, ((0, pad_height), (0, 0), (0, 0)), mode='constant', constant_values=0)

        return img2_resized

def calculate_psnr(img1, img2, data_range=255):
    mse = sk_metrics.mean_squared_error(img1, img2)
    psnr = sk_metrics.peak_signal_noise_ratio(img1, img2, data_range=data_range)
    return psnr


if __name__ == '__main__':
    # Initialization and loading images
    lsb = LSB_Embed()
    background = cv2.imread('LSB/image/lena.bmp')
    watermark = cv2.imread('LSB/image/WaterMark.jpg')
    # print("Background shape:", background.shape)
    # print("Watermark shape:", watermark.shape)

    background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
    watermark = cv2.cvtColor(watermark, cv2.COLOR_BGR2RGB)

    # Original images
    background_backup = background.copy()
    watermark_backup = watermark.copy()

    # Embedding and extracting without attack
    embed_bit = 1
    synthesis = lsb.lsb_embed(background, watermark, embed_bit)
    extract_watermark, extract_background = lsb.lsb_extract(synthesis, embed_bit)
    watermark_resized = lsb.resize_images(watermark, extract_watermark)

    # Calculate SSIM and PSNR before any attack
    ssim_value_before, _ = ssim(cv2.cvtColor(watermark, cv2.COLOR_RGB2GRAY),
                                cv2.cvtColor(watermark_resized, cv2.COLOR_RGB2GRAY), full=True)
    psnr_value_before = calculate_psnr(background, synthesis)

    print(f"Before any attack - PSNR: {psnr_value_before}, SSIM: {ssim_value_before}")

    # Apply and display each attack separately
    attacks = ["gaussian_noise", "salt_and_pepper_noise", "mean_filter", "median_filter", "high_pass_filter", "rotate_image"]

    # Initialize a subplot
    plt.figure(figsize=(15, 15))

    # Display images before any attack
    imgs = [background_backup, synthesis, extract_watermark]
    title = ["Original_Img", "+WaterMark", "Get_WaterMark"]

    for i in range(len(imgs)):
        plt.subplot(3, len(attacks) + 1, i + 1)
        plt.imshow(imgs[i])
        plt.axis("off")
        plt.title(title[i], fontsize=8)  # Set smaller font size

    with open('LSB/txt/result.txt', 'w') as file:
        file.write('Results for LSB\n\n')
        # Apply and display each attack separately
        for idx, attack in enumerate(attacks):
            # Apply the attack
            attacked_image = globals()[attack](synthesis)

            # Extract after the attack
            extract_watermark_attacked, _ = lsb.lsb_extract(attacked_image, embed_bit)

            # Resize images after the attack
            watermark_resized_attacked = lsb.resize_images(watermark, extract_watermark_attacked)

            # Calculate SSIM and PSNR after the attack
            ssim_value_after, _ = ssim(cv2.cvtColor(watermark, cv2.COLOR_RGB2GRAY),
                                    cv2.cvtColor(watermark_resized_attacked, cv2.COLOR_RGB2GRAY), full=True)
            psnr_value_after = calculate_psnr(background, attacked_image)

            print(f"After {attack} - PSNR: {psnr_value_after}, SSIM: {ssim_value_after}")
            file.write(f"After {attack} - PSNR: {psnr_value_after}, SSIM: {ssim_value_after}\n")

            # Display images after the attack
            plt.subplot(3, len(attacks) + 1, (idx + 1) + len(attacks) + 1)
            plt.imshow(attacked_image)
            plt.axis("off")
            plt.title(f"{attack.capitalize()}_Attacked", fontsize=8)  # Set smaller font size

            # Display extracted watermark after the attack
            plt.subplot(3, len(attacks) + 1, (idx + 1) + 2 * len(attacks) + 2)
            plt.imshow(extract_watermark_attacked)
            plt.axis("off")
            plt.title(f"Extracted_Watermark_{attack.capitalize()}", fontsize=6)  # Set smaller font size

        plt.tight_layout()
        plt.show()
