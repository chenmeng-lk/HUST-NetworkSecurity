import math
import random
import cv2
import numpy as np

global dct_encoded_image_file
global DCT_text_len
def string_to_bin(s):
    return ''.join(format(byte, '08b') for byte in s.encode('utf-8'))

# 将二进制字符串转换为字符串（使用UTF-8解码）
def bin_to_string(b):
    try:
        # 将二进制数据按8位切分成字节，然后解码为utf-8字符串
        byte_array = bytearray(int(b[i:i+8], 2) for i in range(0, len(b), 8))
        return byte_array.decode('utf-8')
    except Exception as e:
        return None  # 返回None表示解码失败
    
# DCT 隐写函数
def DCT_yinxie():
    global dct_encoded_image_file
    global DCT_text_len

    print("读取载体图像D:/NetworkSecurity/resources/DCT/DCT.bmp")
    image_path = 'D:/NetworkSecurity/resources/DCT/DCT.bmp'
    y = cv2.imread(image_path, 0)  # 读取灰度图像
    
    row, col = y.shape
    row = int(row / 8)
    col = int(col / 8)

    y1 = y.astype(np.float32)
    Y = cv2.dct(y1)  # 对图像进行 DCT 变换

    while True:  # 添加循环，直到能够成功隐藏信息
        # 用户输入要隐藏的信息
        global tmp
        tmp = input("请输入要嵌入音频的秘密信息(支持中文、英文，数字)例如：\n1人生若只如初见，何事秋风悲画扇。2众里寻他千百度，蓦然回首，那人却在，灯火阑珊处。3Fight\n")
        msg = string_to_bin(tmp)
        count = len(msg)
        DCT_text_len = count
        
        k1, k2 = randrc(row, col, count, 12)
        
        # 如果返回的 k1 和 k2 的长度与要嵌入的信息长度不一致，说明载体图像不够大
        if len(k1) != count or len(k2) != count:
            print("载体图像太小，无法隐藏该信息，请重新输入一个较短的秘密信息。")
        else:
            break  # 如果能成功生成足够的空间，退出循环

    # 嵌入信息
    H = 1
    for i in range(0, count):
        r = (k1[i] - 1) * 8  # 恢复行索引
        c = (k2[i] - 1) * 8  # 恢复列索引
        # 处理嵌入信息
        if msg[i] == '0':
            if Y[r + 4, c + 1] > Y[r + 3, c + 2]:
                Y[r + 4, c + 1], Y[r + 3, c + 2] = swap(Y[r + 4, c + 1], Y[r + 3, c + 2])
        else:
            if Y[r + 4, c + 1] < Y[r + 3, c + 2]:
                Y[r + 4, c + 1], Y[r + 3, c + 2] = swap(Y[r + 4, c + 1], Y[r + 3, c + 2])
        
        # 根据嵌入后的信息调整 DCT 系数
        if Y[r + 4, c + 1] > Y[r + 3, c + 2]:
            Y[r + 3, c + 2] = Y[r + 3, c + 2] - H
        else:
            Y[r + 4, c + 1] = Y[r + 4, c + 1] - H

    y2 = cv2.idct(Y)  # 得到嵌入信息后的图像
    dct_encoded_image_file = 'D:/NetworkSecurity/resources/DCT/DCT_embedded.bmp'
    cv2.imwrite(dct_encoded_image_file, y2)  # 保存图像
    print(f"图像隐写已完成,隐写后的图像保存为{dct_encoded_image_file}")

# DCT 提取函数
def DCT_tiqu():
    count = int(DCT_text_len)
    print(f"开始提取，隐写图像路径：{dct_encoded_image_file}")
    dct_img = cv2.imread(dct_encoded_image_file, 0)
    if dct_img is None:
        print("图像加载失败，请检查文件路径和格式")
        return

    y = dct_img
    y1 = y.astype(np.float32)
    Y = cv2.dct(y1)

    row, col = y.shape
    row = int(row / 8)
    col = int(col / 8)
    
    k1, k2 = randrc(row, col, count, 12)
    
    # 提取并回写信息
    b = ""
    for i in range(0, count):
        r = (k1[i]-1) * 8  # 恢复行索引
        c = (k2[i]-1) * 8  # 恢复列索引
        if Y[r + 4, c + 1] < Y[r + 3, c + 2]:
            b += '0'
        else:
            b += '1'
    restored_text = bin_to_string(b)
    if restored_text is None: 
        print("解码失败，无法显示解码结果，隐藏信息过长导致编码出错。")
    else:
        print(f"提取的文本信息: {restored_text}")  # 打印提取的信息
    if restored_text == tmp:
        print("提取的信息与原始信息匹配，信息隐藏成功！")
    else:
        print("提取的信息与原始信息不匹配，信息隐藏失败！")

#randrc 函数会根据输入的 count（要隐藏的信息的二进制位数）生成随机的
# DCT系数位置 ，这些位置对应于图像的DCT变换后的矩阵Y中的系数
def randrc(m, n, count, key):
    # m, n = matrix.shape
    q1 = int(m * n / count) + 1
    q2 = q1 - 2
    if q2 == 0:
        print('载体太小，不能将秘密信息隐藏进去!')
        return [], []

    # 设置随机种子，确保可重复生成随机序列
    random.seed(key)
    
    a = [0] * count  # a 是 list
    for i in range(count):
        a[i] = random.random()
    # 初始化
    row = [0] * count
    col = [0] * count
    # 记录已生成的 (row, col) 对
    seen_positions = set()

    # 计算 row 和 col
    r = 0
    c = 0
    row[0] = r
    col[0] = c
    seen_positions.add((r, c))  # 添加第一个位置到 set

    for i in range(1, count):
        # 根据 a[i] 的值来决定是选择大区间 q1 还是小区间 q2 来生成新的列索引 c
        if a[i] >= 0.5:
            c = c + q1
        else:
            c = c + q2
        if c > n:  # 如果列超出了最大值，则回绕
            k = c % n  # 列取模
            r = r + int((c - k) / n)  # 根据列回绕更新行
            if r > m:  # 如果行超出最大值，则退出
                print('载体太小不能将秘密信息隐藏进去!')
                return [], []
            c = k  # 更新列位置
            if c == 0:  # 如果列为 0，则调整列为 1
                c = 1       
        # 如果该 (r, c) 已经出现过，则重新生成
        while (r, c) in seen_positions:
            if a[i] >= 0.5:
                c = c + q1
            else:
                c = c + q2
            
            if c > n:
                k = c % n
                r = r + int((c - k) / n)
                if r > m:
                    print('载体太小不能将秘密信息隐藏进去!')
                    return [], []
                
                c = k
                if c == 0:
                    c = 1
        
        # 更新 row, col, 并将新位置添加到 seen_positions
        row[i] = r
        col[i] = c
        seen_positions.add((r, c))
    
    return row, col

def plus(str):
	return str.zfill(8)

def get_key(file_path):
    # 打开文件并读取内容
    with open(file_path, "rb") as f:
        content = f.read()
    return ''.join(format(byte, '08b') for byte in content.encode('utf-8'))

def swap(a, b):
    # 用于交换两个值的简单函数
    return b, a
def toasc(strr):
	return int(strr, 2)

# 计算峰值信噪比（PSNR）
def calculate_psnr(image_path1, image_path2):
    # 打开图像
    img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)

    # 计算均方误差（MSE）
    mse = np.mean((img1 - img2) ** 2)

    # 若MSE为0，PSNR为无穷大，表示两张图完全相同
    if mse == 0:
        return float('inf')

    # 获取图像的最大像素值，通常为255（灰度图像）
    max_pixel = 255.0

    # 计算PSNR
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

# 计算结构相似度（SSIM）
def calculate_ssim(image_path1, image_path2):
    # 读取图像并转换为灰度图
    img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)

    # 确保两个图像大小一致
    if img1.shape != img2.shape:
        raise ValueError("输入图像必须具有相同的尺寸")

    # 计算均值、方差和协方差
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    window = np.ones((3, 3)) / 9  # 简单的3x3均值滤波器

    mu1 = cv2.filter2D(img1.astype(np.float64), -1, window)
    mu2 = cv2.filter2D(img2.astype(np.float64), -1, window)

    sigma1 = cv2.filter2D(img1.astype(np.float64) ** 2, -1, window) - mu1 ** 2
    sigma2 = cv2.filter2D(img2.astype(np.float64) ** 2, -1, window) - mu2 ** 2
    sigma12 = cv2.filter2D(img1.astype(np.float64) * img2.astype(np.float64), -1, window) - mu1 * mu2

    # 计算 SSIM
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2))
    ssim_value = ssim_map.mean()
    
    return ssim_value
# 计算并显示 PSNR 和 SSIM
def evaluate_images():
    original_image_path = 'D:/NetworkSecurity/resources/DCT/DCT.bmp'  # 原图路径
    embedded_image_path = 'D:/NetworkSecurity/resources/DCT/DCT_embedded.bmp'  # 嵌入水印后的图像路径
    
    # 计算PSNR
    psnr_value = calculate_psnr(original_image_path, embedded_image_path)
    print(f"峰值信噪比PSNR: {psnr_value} dB")
    
    # 计算SSIM
    ssim_value = calculate_ssim(original_image_path, embedded_image_path)
    print(f"结构相似度SSIM: {ssim_value}")


# Main 函数执行
def main():
    DCT_yinxie()
    DCT_tiqu()
    evaluate_images()
# 执行main函数
if __name__ == "__main__":
    main()