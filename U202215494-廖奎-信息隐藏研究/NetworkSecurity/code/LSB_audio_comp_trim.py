import wave
import numpy as np

def bin_to_string(b):
    try:
        # 将二进制数据按8位切分成字节，然后解码为utf-8字符串
        byte_array = bytearray(int(b[i:i+8], 2) for i in range(0, len(b), 8))
        return byte_array.decode('utf-8')
    except Exception as e:
        return None  # 返回None表示解码失败
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
# 压缩音频（通过降低采样率）
def compress_audio(input_path, output_path, new_sample_rate=22050):
    with wave.open(input_path, 'rb') as input_audio:
        # 获取音频参数
        params = input_audio.getparams()
        num_channels = params.nchannels  # 声道数
        sampwidth = params.sampwidth      # 每个采样的字节数
        framerate = params.framerate      # 原始采样率
        num_frames = params.nframes       # 总帧数
        
        # 读取音频数据
        audio_data = input_audio.readframes(num_frames)
        
    # 将音频数据转为numpy数组
    audio_array = np.frombuffer(audio_data, dtype=np.int16)
    
    # 计算新的帧数
    new_num_frames = int(num_frames * new_sample_rate / framerate)
    
    # 重新采样音频数据（降采样）
    compressed_audio_array = audio_array[::int(framerate / new_sample_rate)]
    
    # 将音频数据保存到新的wave文件
    with wave.open(output_path, 'wb') as output_audio:
        output_audio.setparams((num_channels, sampwidth, new_sample_rate, new_num_frames, 'NONE', 'not compressed'))
        output_audio.writeframes(compressed_audio_array.tobytes())
        
    print(f"音频已压缩并保存为 {output_path}")

# 裁剪音频（只保留前1/3部分）
def trim_audio(input_path, output_path):
    print("裁剪音频中")
    with wave.open(input_path, 'rb') as input_audio:
        # 获取音频参数
        params = input_audio.getparams()
        num_channels = params.nchannels  # 声道数
        sampwidth = params.sampwidth      # 每个采样的字节数
        framerate = params.framerate      # 原始采样率
        num_frames = params.nframes       # 总帧数
        
        # 计算裁剪后的帧数（只保留前1/3）
        new_num_frames = num_frames // 3
        
        # 读取音频数据并裁剪
        audio_data = input_audio.readframes(new_num_frames)
        
    # 将裁剪后的音频数据保存到新的wave文件
    with wave.open(output_path, 'wb') as output_audio:
        output_audio.setparams((num_channels, sampwidth, framerate, new_num_frames, 'NONE', 'not compressed'))
        output_audio.writeframes(audio_data)
        

if __name__ == "__main__":
    output_audio_path = 'D:/NetworkSecurity/resources/LSB/LSB_embedded.wav'
    secret_message = extract_data(output_audio_path)
    print(f"读取载体音频LSB_embedded.wav，其中加密信息：\n{secret_message}\n")

    # 压缩音频
    print("压缩音频中")
    compressed_audio_path = 'D:/NetworkSecurity/resources/LSB/LSB_compressed.wav'
    compress_audio(output_audio_path, compressed_audio_path, new_sample_rate=22050)
    # 从压缩后的音频中提取信息
    extracted_message = extract_data(compressed_audio_path)
    if(extracted_message is None): 
        print("解码失败，无法显示解码结果。")
    else:
        print(f"从压缩后的音频中提取出的信息: {extracted_message}")
    # 比较提取的消息和原始消息
    if extracted_message == secret_message:
        print("压缩后提取的信息与原始信息匹配，信息隐藏成功！")
    else:
        print("压缩后提取的信息与原始信息不匹配，信息隐藏失败！\n")

    # 裁剪音频（只保留前1/3部分）
    trimmed_audio_path = 'D:/NetworkSecurity/resources/LSB/LSB_trimmed.wav'
    trim_audio(output_audio_path, trimmed_audio_path)
    print(f"成功裁剪载体音频，将前1/3部分保存到{trimmed_audio_path}")
    # 从裁剪后的音频中提取信息
    extracted_message_trimmed = extract_data(trimmed_audio_path)
    if(extracted_message_trimmed is None): 
        print("解码失败，无法显示解码结果。")
    else:
        print(f"从裁剪后的音频中提取出的信息: {extracted_message_trimmed}")
    
    # 比较裁剪后提取的信息与原始信息
    if extracted_message_trimmed == secret_message:
        print("裁剪后提取的信息与原始信息匹配，信息隐藏成功！")
    else:
        print("裁剪后提取的信息与原始信息不匹配，信息隐藏失败！")