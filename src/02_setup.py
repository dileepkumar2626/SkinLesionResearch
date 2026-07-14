from google.colab import drive
drive.mount('/content/drive')
from google.colab import files
files.upload()
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
unzip -oq skin-cancer-mnist-ham10000.zip -d /content/drive/MyDrive/ham10000_data
import os
print(os.listdir("/content/drive/MyDrive/ham10000_data"))