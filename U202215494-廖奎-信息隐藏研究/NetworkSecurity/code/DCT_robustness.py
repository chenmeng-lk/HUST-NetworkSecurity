from PIL import Image, ImageDraw
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

    # 用户输入要隐藏的信息
    global tmp
    tmp = input("请输入要嵌入音频的秘密信息(支持中文、英文，数字)例如：\n1人生若只如\
初见，何事秋风悲画扇。2众里寻他千百度，蓦然回首，那人却在，灯火阑珊处。3Fight\n")
    msg=string_to_bin(tmp);
    count = len(msg)
    DCT_text_len = count
    
    k1, k2 = randrc(row, col, count, 12)
    
    # 嵌入信息
    H = 1
    for i in range(0, count):
        r = (k1[i]-1) * 8  # 恢复行索引
        c = (k2[i]-1) * 8  # 恢复列索引
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

    # 压缩为JPEG格式，设置压缩质量（如80%）
    dct_encoded_image_file2 = 'D:/NetworkSecurity/resources/DCT/DCT_embedded_compressed.jpg'
    cv2.imwrite(dct_encoded_image_file2, y2, [cv2.IMWRITE_JPEG_QUALITY, 80])  # 保存为JPEG，质量设置为80
    print(f"把隐写后的图像压缩，压缩质量80%, 保存为{dct_encoded_image_file2}")

# 计算汉明距离的函数
def hamming_distance(str1, str2):
    # 计算两个二进制字符串的汉明距离
    return sum(c1 != c2 for c1, c2 in zip(str1, str2))

#  DCT 提取函数
def DCT_tiqu(dct_encoded_image_file_path):
    count = int(DCT_text_len)
    print(f"开始提取，隐写图像路径：{dct_encoded_image_file_path}")
    dct_img = cv2.imread(dct_encoded_image_file_path, 0)
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
        print("隐藏信息被更改，解码失败，无法显示解码结果。")
    else:
        print(f"提取的文本信息: {restored_text}")  # 打印提取的信息

    # 计算提取的信息与原始信息的相似度（基于二进制流的汉明距离）
    # 获取原始隐藏信息的二进制流
    original_binary = string_to_bin(tmp)    
    # 计算汉明距离
    distance = hamming_distance(original_binary, b)
        
    # 计算相似度：相似度 = 1 - 汉明距离 / 二进制长度
    similarity = 1 - distance / len(original_binary)
    print(f"提取信息与原始信息的相似度: {similarity:.4f}")

    # 检查提取的信息是否与原始信息匹配
    if restored_text == tmp:
        print("提取的信息与原始信息匹配，信息隐藏成功！\n")
    else:
        print("提取的信息与原始信息不匹配，信息隐藏失败！\n")


#randrc 函数会根据输入的 count（要隐藏的信息的二进制位数）生成随机的
#DCT系数位置，这些位置对应于图像的DCT变换后的矩阵Y中的系数
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
    # 将每个字节转换为二进制，并拼接成一个长的二进制字符串
    binary_str = ''.join(format(byte, '08b') for byte in content)

    # 返回二进制字符串
    return binary_str

def swap(a, b):
    # 用于交换两个值的简单函数
    return b, a
def toasc(strr):
	return int(strr, 2)
def cover():
    image = Image.open('D:/NetworkSecurity/resources/DCT/DCT_embedded.bmp')
    # 创建一个可以在图片上绘制的对象
    draw = ImageDraw.Draw(image)
    # 定义遮挡区域的坐标 (左上角x, 左上角y, 右下角x, 右下角y)
    box = (100, 100, 300, 300)
    # 绘制一个白色矩形
    draw.rectangle(box, fill="white")
    # 保存修改后的图片
    image.save('D:/NetworkSecurity/resources/DCT/DCT_cover.bmp')
    print(f"DCT_embedded.bmp遮挡成功，生成图片DCT_cover.bmp")

# 旋转图像并保存
def rotate_image(image_path, angle):
    print(f"正在旋转图像 {angle}°")
    image = Image.open(image_path)
    rotated_image = image.rotate(angle, expand=True)  # 旋转并扩展图像以避免裁剪
    rotated_image_path = image_path.replace('.bmp', f'_rotated.bmp')
    rotated_image.save(rotated_image_path)
    return rotated_image_path
# 执行main函数
if __name__ == "__main__":
    print("以下是DCT算法抗攻击性检测")
    DCT_yinxie()
    print("\n从未经处理的载体图像DCT_embedded.bmp中提取隐藏信息")
    DCT_tiqu(dct_encoded_image_file)
    cover()
    path='D:/NetworkSecurity/resources/DCT/DCT_cover.bmp'
    DCT_tiqu(path)
    print("从压缩后的图像DCT_embedded_compressed.jpg中提取隐藏信息")
    DCT_tiqu('D:/NetworkSecurity/resources/DCT/DCT_embedded_compressed.jpg')
    # 旋转图像并提取隐藏信息
    rotated_image_path = rotate_image(dct_encoded_image_file, 1)
    print(f"从旋转后的图像{rotated_image_path}中提取隐藏信息")
    DCT_tiqu(rotated_image_path)


