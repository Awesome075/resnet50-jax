import urllib.request
import os

url = "https://huggingface.co/microsoft/resnet-50/resolve/main/flax_model.msgpack"
filename = "flax_model.msgpack"

if not os.path.exists(filename):
	print(f"Downloading {filename}...")
	urllib.request.urlretrieve(url, filename)
	print("Download complete.")