from PIL import Image, ImageDraw
import numpy as np

def extract_watermark(image_path, original_image_size, output_path):
    # 读取压缩后的图片
    image = Image.open(image_path)
    image_data = np.array(image)
    
    # 获取原图的宽度和高度
    width, height = original_image_size

    # 获取 B 通道的最低有效位作为水印
    watermark_data = image_data[:, :, 2] & 1  # 获取 B 通道的最低有效位
    watermark_image = Image.new('1', (width, height))

    # 使用putdata方法将二维数组的数据放入图片中
    watermark_image.putdata(watermark_data.flatten())
    watermark_image.save(output_path)

def compress_image(input_path, output_path, quality=30):
    """
    压缩图像并保存，使用JPEG压缩
    param quality: 压缩质量，数值越小图像越模糊
    """
    image = Image.open(input_path)
    image = image.convert("RGB")  # 确保是RGB模式
    image.save(output_path, "JPEG", quality=quality)
    print(f"图像压缩成功，保存为 {output_path}")

if __name__ == "__main__":
    # 初始载体图像路径
    input_image_path = 'D:/NetworkSecurity/resources/LSB/LSB_embedded.png'

    # 读取图像并获取尺寸
    image = Image.open(input_image_path)
    original_image_size = image.size
    
    # 设置压缩后的载体图像保存路径
    compressed_image_path = 'D:/NetworkSecurity/resources/LSB/LSB_compressed.jpg'
    
    # 对载体图像进行压缩
    compress_image(input_image_path, compressed_image_path, quality=30)

    # 设置提取水印图像保存路径
    output_path_extract = "D:/NetworkSecurity/resources/LSB/LSB_compressed_watermark.png"

    # 从压缩后的图像中提取水印
    extract_watermark(compressed_image_path, original_image_size, output_path_extract)
    print(f"从压缩后的图片提取的水印已保存为 {output_path_extract}")

    print("操作完成！")
