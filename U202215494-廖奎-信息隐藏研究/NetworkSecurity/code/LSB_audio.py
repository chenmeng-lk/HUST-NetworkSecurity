import wave
import numpy as np
import os

# 将字符串转换为二进制字符串（使用UTF-8编码）
def string_to_bin(s):
    return ''.join(format(byte, '08b') for byte in s.encode('utf-8'))

# 将二进制字符串转换为字符串（使用UTF-8解码）
def bin_to_string(b):
    # 将二进制数据按8位切分成字节，然后解码为utf-8字符串
    byte_array = bytearray(int(b[i:i+8], 2) for i in range(0, len(b), 8))
    return byte_array.decode('utf-8')

# 嵌入数据到音频文件
def embed_data(audio_path, data, output_path):
    # 打开音频文件
    with wave.open(audio_path, 'rb') as audio:
        params = audio.getparams()
        frames = audio.readframes(params.nframes)
    
    # 转换音频帧为numpy数组
    audio_array = np.frombuffer(frames, dtype=np.int16)

    # 创建一个可以修改的副本
    audio_array = audio_array.copy()
    # 将数据转换为二进制字符串
    data_bin = string_to_bin(data)

    end_symbol = '1011011101111011111011110111011010'
    data_bin += end_symbol  # 在数据末尾加上结束符

    data_length = len(data_bin)
    
    # 每个音频样本占两个字节（16位），所以数据嵌入的最大长度是音频帧数量
    if data_length > len(audio_array):
        raise ValueError("数据太长，无法嵌入音频文件。")

    # 嵌入数据到音频的最低有效位
    for i in range(data_length):
        # 获取数据的当前位
        bit = int(data_bin[i])
        # 将音频数组的当前帧的最低有效位替换为数据位
        audio_array[i] = (audio_array[i] & ~1) | bit
    
    # 保存嵌入数据后的音频
    with wave.open(output_path, 'wb') as output_audio:
        output_audio.setparams(params)
        output_audio.writeframes(audio_array.tobytes())

# 从音频文件中提取隐藏的数据
def extract_data(audio_path):
    with wave.open(audio_path, 'rb') as audio:
        params = audio.getparams()
        frames = audio.readframes(params.nframes)
    
    # 转换音频帧为numpy数组
    audio_array = np.frombuffer(frames, dtype=np.int16)

    # 提取音频样本的最低有效位
    extracted_bits = []
    for sample in audio_array:
        extracted_bits.append(sample & 1)

    # 将二进制位转换为字符串
    extracted_bin = ''.join(str(bit) for bit in extracted_bits)
    # 查找结束标志
    end_index = extracted_bin.find('1011011101111011111011110111011010')  # 查找结束符
    if end_index != -1:
        extracted_bin = extracted_bin[:end_index]
    return bin_to_string(extracted_bin)

# 计算均方误差（MSE）
def calculate_mse(original_audio_path, modified_audio_path):
    with wave.open(original_audio_path, 'rb') as audio:
        params = audio.getparams()
        frames = audio.readframes(params.nframes)
    original_audio_array = np.frombuffer(frames, dtype=np.int16)

    with wave.open(modified_audio_path, 'rb') as audio:
        frames = audio.readframes(params.nframes)
    modified_audio_array = np.frombuffer(frames, dtype=np.int16)

    # 计算均方误差
    mse = np.mean((original_audio_array - modified_audio_array) ** 2)
    return mse
# 计算PSNR（基于MSE）
def calculate_psnr(mse):
    if mse == 0:
        return 100  # 两者完全相同，PSNR值为100

    max_pixel_value = 32767  # 16位音频最大值
    psnr_value = 10 * np.log10((max_pixel_value ** 2) / mse)
    return psnr_value

# 计算相似度百分比
def calculate_similarity_percentage(psnr_value):
    # 设定一个阈值范围，将PSNR值映射到相似度百分比
    if psnr_value >= 40:
        return 100  # 高于40 dB认为完全相似
    elif psnr_value >= 30:
        return 90  # 30-40 dB认为相似度90%
    elif psnr_value >= 20:
        return 75  # 20-30 dB认为相似度75%
    elif psnr_value >= 10:
        return 50  # 10-20 dB认为相似度50%
    else:
        return 0  # 低于10 dB认为几乎不相似
# 主函数
def main():
    # 获取用户输入的秘密消息（可以输入中文或英文）
    secret_message = input("请输入要嵌入音频的秘密信息(支持中文、英文、数字、部分符号)示例：\n\
1.huster。2.大鹏一日同风起，扶摇直上九万里。3.青青子衿，悠悠我心。\n")  # 动态输入

    input_audio_path = 'D:/NetworkSecurity/resources/LSB/LSB.wav'  # 使用 .wav 格式
    output_audio_path = 'D:/NetworkSecurity/resources/LSB/LSB_embedded.wav'

    # 确保文件存在
    if not os.path.exists(input_audio_path):
        raise FileNotFoundError(f"音频文件 {input_audio_path} 未找到")
    
    # 将信息嵌入到音频文件
    embed_data(input_audio_path, secret_message, output_audio_path)
    print(f"信息成功嵌入到 {output_audio_path}")
    
    # 从音频中提取信息
    extracted_message = extract_data(output_audio_path)
    print(f"从音频中提取出的信息: {extracted_message}")
    
    # 比较提取的消息和原始消息
    if extracted_message == secret_message:
        print("提取的信息与原始信息匹配，信息隐藏成功！")
    else:
        print("提取的信息与原始信息不匹配，信息隐藏失败！")
    # 计算MSE
    mse = calculate_mse(input_audio_path, output_audio_path)
    psnr=calculate_psnr(mse)
    # 计算相似度百分比
    similarity_percentage = calculate_similarity_percentage(psnr)
    print(f"载体音频与原音频相似度百分比(近似值): {similarity_percentage}%")
if __name__ == "__main__":
    main()
