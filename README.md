# ResNet-50 v1.5 in JAX/Flax

A clean, modular reimplementation of Microsoft's ResNet-50 v1.5 model using JAX and the Flax `nnx` API. 

This repository provides a ground-up implementation of the architecture, along with a custom weight-mapping utility to load pre-trained Hugging Face weights directly into the custom `nnx` state.

## Features

- **Flax `nnx` API:** Built using the stateful neural network API for Flax.
- **Native Weight Conversion:** Maps official Hugging Face `.msgpack` weights to custom JAX arrays, handling key mapping and model state updates directly.

## Repository Structure

- `modeling.py`: Ground-up ResNet-50 v1.5 architecture (convolutional stem, residual bottleneck stages, and linear classification head).
- `download_weights.py`: Utility script to download pre-trained weights from the Hugging Face Hub.
- `conversion.py`: Core utility to load Hugging Face and weights into the `nnx` model state.
- `inference.py`: Image classification script evaluating sample input against ImageNet-1K labels.
- `requirements.txt`: Project dependencies.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Awesome075/resnet-jax.git
   cd resnet-flax
   ```
2. Create a virtual environment and install dependencies:
   
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate  
   # On macOS/Linux
   source venv/bin/activate

   pip install -r requirements.txt
   ```

3. **Download Weights:**  

   ```bash
   python download_weights.py
 
   ``` 

## Usage

Run the inference script to evaluate an image and output top predicted ImageNet-1K class:

```bash
python inference.py
```

