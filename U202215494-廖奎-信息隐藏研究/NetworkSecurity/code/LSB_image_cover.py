from PIL import Image, ImageDraw
import numpy as np
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

if __name__ == "__main__":
    # 打开图片
    image = Image.open('D:/NetworkSecurity/resources/LSB/LSB_embedded.png')
    # 创建一个可以在图片上绘制的对象
    draw = ImageDraw.Draw(image)
    # 定义遮挡区域的坐标 (左上角x, 左上角y, 右下角x, 右下角y)
    box = (50, 50, 200, 200)
    # 绘制一个白色矩形
    draw.rectangle(box, fill="white")
    # 保存修改后的图片
    image.save('D:/NetworkSecurity/resources/LSB/LSB_cover.png')
    print(f"LSB_embedded.png遮挡成功，生成图片LSB_cover.png")
    output_path_embed='D:/NetworkSecurity/resources/LSB/LSB_cover.png'
    image = Image.open(output_path_embed)
    color_image_size=image.size
    output_path_extract = "D:/NetworkSecurity/resources/LSB/LSB_coverd_watermark.png"
    extract_watermark(output_path_embed, color_image_size, output_path_extract)
    print("从图片LSB_cover.png提取的水印图片已保存为LSB_coverd_watermark.png")
    print("操作完成！")
