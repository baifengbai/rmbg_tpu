import requests
import base64

def encode_image_to_base64(image_path):
    """将图像文件编码为base64字符串"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def send_remove_bg_request(image_base64):
    """发送POST请求到FastAPI服务器以移除图像背景"""
    url = 'http://localhost:8000/remove-bg/'
    headers = {'Content-Type': 'application/json'}
    data = {'image_base64': image_base64}

    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        # 处理成功，保存返回的图像数据
        with open('output_image.png', 'wb') as out_image:
            out_image.write(response.content)
        print("背景移除成功，输出图像已保存为 output_image.png")
    else:
        print("背景移除失败：", response.text)

if __name__ == "__main__":
    image_path = 'example_input.jpg'
    image_base64 = encode_image_to_base64(image_path)
    send_remove_bg_request(image_base64)

