from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

# 生成黑白水印图片，输入文本会生成一个合适大小的水印，图片大小与彩色图像相同
def create_black_white_watermark(color_image_size, text="Watermark", max_font_size=25):
    try:
        font = ImageFont.truetype("arial.ttf", max_font_size)
    except IOError:
        font = ImageFont.load_default()

    # 获取彩色图片的大小
    width, height = color_image_size
    
    # 创建黑色背景图片，尺寸与彩色图像相同
    image = Image.new('1', (width, height), color=0)  # '1'表示黑白图像，背景设为黑色
    draw = ImageDraw.Draw(image)

    # 计算文字的大小并确保适应图片大小
    font_size = max_font_size

    # 将文本分割为多行，每行最多25个字符
    max_chars_per_line = 25
    lines = [text[i:i + max_chars_per_line] for i in range(0, len(text), max_chars_per_line)]
    
    # 计算总的文本高度，包括行间距
    line_height = font_size  # 行间距为字体大小
    total_text_height = line_height * len(lines)

    # 计算起始位置，使得文本垂直居中
    start_y = (height - total_text_height) // 2

    # 在图片上绘制每一行文字
    for i, line in enumerate(lines):
        text_width = draw.textbbox((0, 0), line, font=font)[2]
        x_position = (width - text_width) // 2  # 水平居中
        y_position = start_y + i * line_height  # 垂直排列每一行

        draw.text((x_position, y_position), line, fill=255, font=font)  # fill=255表示白色文字

    return image


# 嵌入水印到彩色图片的 B 通道的最低有效位
def embed_watermark(image_path, watermark_image_path, output_path):
    # 打开彩色图片并转为 RGB 模式
    color_image = Image.open(image_path).convert("RGB")
    color_data = np.array(color_image)

    # 读取水印图像
    watermark_image = Image.open(watermark_image_path).convert('L')  # 转换为灰度图像
    # 确保水印图片的尺寸与彩色图片匹配
    watermark_image = watermark_image.resize(color_image.size)
    watermark_data = np.array(watermark_image)

    # 二值化处理：将所有非零的像素值设置为1（白色），将零像素值设置为0（黑色）
    watermark_data = (watermark_data > 128).astype(int)  # 这里的阈值128可以根据需求调整

    # 将水印数据嵌入到 B 通道的最低有效位
    for i in range(color_data.shape[0]):
        for j in range(color_data.shape[1]):
            r, g, b = color_data[i, j]

            # 修改 B 通道的最低有效位
            b = (b & 0b11111110) | watermark_data[i, j]

            # 更新像素
            color_data[i, j] = (r, g, b)

    # 保存带有水印的图片
    modified_image = Image.fromarray(color_data)
    modified_image.save(output_path)

# 提取嵌入图片中的水印信息
def extract_watermark(image_path, original_image_size, output_path):
    # 读取嵌入水印后的图片
    image = Image.open(image_path)
    image_data = np.array(image)
    # 获取原图的宽度和高度
    width , height= original_image_size

    watermark_data = image_data[:, :, 2] & 1  # 获取 B 通道的最低有效位
    watermark_image = Image.new('1', (width, height))

    # 使用putdata方法将二维数组的数据放入图片中
    # 由于putdata需要一维数组，我们先将watermark_data转换为一维
    watermark_image.putdata(watermark_data.flatten())
    watermark_image.save(output_path)

# 计算PSNR
def calculate_psnr(image_path1, image_path2):

    #计算峰值信噪比（PSNR）。

    # 读取图像
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)

    # 确保两个图像大小一致
    if img1.shape != img2.shape:
        raise ValueError("输入图像必须具有相同的尺寸")

    # 计算均方误差 (MSE)
    mse = np.mean((img1 - img2) ** 2)
    
    if mse == 0:
        return 100  # 如果两个图像完全相同，PSNR值为 100

    # 计算PSNR
    PIXEL_MAX = 255.0
    psnr_value = 10 * np.log10((PIXEL_MAX ** 2) / mse)
    
    return psnr_value


# 计算SSIM（简化版，基于亮度、对比度和结构信息）
def calculate_ssim(image_path1, image_path2):
    #计算结构相似度（SSIM），使用 OpenCV 和 NumPy 实现。
   
    # 读取图像并转为灰度
    img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)

    # 计算均值
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)

    # 计算方差
    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)

    # 计算协方差
    sigma12 = np.cov(img1.flatten(), img2.flatten())[0, 1]

    # 设置常量
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    # 计算SSIM
    ssim_value = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_value

    
    

# 主程序
if __name__ == "__main__":
    # 1. 读取彩色图片，获取其大小
    image_path = "D:/NetworkSecurity/resources/LSB/LSB.png"  # 彩色图片路径
    color_image = Image.open(image_path)
    color_image_size = color_image.size

    # 获取用户输入的水印文本
    watermark_text = input("请输入水印字符串(英文)(例如 Meet at the park): ")
    # 生成黑白水印图片，图片大小与彩色图片一致
    watermark_image = create_black_white_watermark(color_image_size, text=watermark_text)

    # 保存黑白水印图片
    watermark_image_path = "D:/NetworkSecurity/resources/LSB/LSB_create_watermark.png"
    watermark_image.save(watermark_image_path)
    print(f"黑白水印图片已保存到 {watermark_image_path}")

    # 3. 嵌入水印到彩色图片并保存为 LSB2.png
    output_path_embed = "D:/NetworkSecurity/resources/LSB/LSB_embedded.png"
    embed_watermark(image_path, watermark_image_path, output_path_embed)
    print(f"嵌入水印后彩色图片已保存到 {output_path_embed}")

    # 4. 提取水印并保存为 LSB_get_watermark.png
    output_path_extract = "D:/NetworkSecurity/resources/LSB/LSB_get_watermark.png"
    extract_watermark(output_path_embed, color_image_size, output_path_extract)
    print(f"提取的水印图片已保存到 {output_path_extract}")
    # 计算PSNR和SSIM
    psnr_value = calculate_psnr(image_path, output_path_embed)
    ssim_value = calculate_ssim(image_path, output_path_embed)

    print(f"PSNR值: {psnr_value} dB")
    print(f"SSIM值: {ssim_value}")

    print("操作完成！")
