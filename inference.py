import requests
import jax.numpy as jnp
from PIL import Image
from flax import nnx
from transformers import AutoImageProcessor, AutoConfig
from modeling import Config, ResNet
from conversion import load_flax_weights
import jax

rngs = nnx.Rngs(42)
config = Config()
model = ResNet(config, rngs)
model = load_flax_weights(model)

image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
url = "https://images.unsplash.com/photo-1543466835-00a7907e9de1"  # Dog (Beagle) picture
image = Image.open(requests.get(url, stream=True).raw)

input_ = image_processor(images=image, return_tensors="np")

#(B,C,H,W) (pytorch Compatible) -> (B,H,W,C) (jax Compatible)
pixel_values = jnp.transpose(input_["pixel_values"],(0,2,3,1))

image_tensor = jnp.array(pixel_values)

logits = model(image_tensor)
probabilities = jax.nn.softmax(logits, axis=-1)

predicted_id = int(jnp.argmax(probabilities, axis=-1).item())
confidence = float(probabilities[0, predicted_id].item())

hf_config = AutoConfig.from_pretrained("microsoft/resnet-50")
predicted_class = hf_config.id2label[predicted_id]

print(f"\nResult:")
print(f"Class ID:   {predicted_id}")
print(f"Prediction: {predicted_class}")
print(f"Confidence: {confidence * 100:.2f}%")
