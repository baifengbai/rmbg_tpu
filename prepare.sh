#!/bin/bash
# 确保脚本在出错时终止执行
set -e

# 安装Python依赖
echo "Installing Python dependencies..."
pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 下载模型文件
echo "Downloading model files..."
python3 -m dfss --url=open@sophgo.com:/aigc/models.zip

# 检查并安装unzip工具
if ! command -v unzip &> /dev/null
then
    echo "unzip could not be found, installing..."
    sudo apt-get install unzip
fi

# 解压模型文件
echo "Unzipping model files..."
unzip models.zip -d ./models


# 检查模型文件是否解压成功
if [ -d "./models/rmbg.bmodel" ]; then
  echo "Bmodel ready."
else
  echo "Bmodel is not ready, check the unzip process."
  exit 1
fi

echo "Environment ready, starting background removal!"
