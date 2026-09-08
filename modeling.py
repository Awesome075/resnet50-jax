import jax
import jax.numpy as jnp
from flax import nnx
import jax.random as random

@nnx.dataclass
class Config:
	num_classes: int = 1000
	stage_sizes: tuple[int] = (3,4,6,3)

class Bottleneck(nnx.Module):
	def __init__(self, in_channels:int, out_channels:int, strides:int, downsample = None, *, rngs:nnx.Rngs):
		self.conv0 = nnx.Conv(in_channels, out_channels, kernel_size=(1,1), strides=1, use_bias=False, rngs=rngs)
		self.bn0 = nnx.BatchNorm(out_channels, use_running_average=True, rngs=rngs)
		self.conv1 = nnx.Conv(out_channels, out_channels, kernel_size=(3,3), strides=strides, padding=1, use_bias=False, rngs=rngs)
		self.bn1 = nnx.BatchNorm(out_channels, use_running_average=True, rngs=rngs)
		self.conv2 = nnx.Conv(out_channels, out_channels * 4, kernel_size=(1,1), padding=0, use_bias=False, rngs=rngs)
		self.bn2 = nnx.BatchNorm(out_channels * 4, use_running_average=True, rngs=rngs)
		self.downsample = downsample

	def __call__(self, x:jnp.ndarray):
		identity = x

		x = self.conv0(x)
		x = self.bn0(x)
		x = nnx.relu(x)		

		x = self.conv1(x)
		x = self.bn1(x)
		x = nnx.relu(x)

		x = self.conv2(x)
		x = self.bn2(x)

		if self.downsample is not None:
			identity = self.downsample(identity)

		return nnx.relu(x + identity)

class Downsample(nnx.Module):
	def __init__(self, in_channels:int, out_channels:int, strides:int, rngs:nnx.Rngs):
		self.conv = nnx.Conv(in_channels, out_channels, kernel_size=(1,1), strides=strides, use_bias=False, rngs=rngs)
		self.bn = nnx.BatchNorm(out_channels, use_running_average=True, rngs=rngs)

	def __call__(self, x:jnp.ndarray):
		x = self.conv(x)
		x = self.bn(x)

		return x


class Stem(nnx.Module):
	def __init__(self, rngs:nnx.Rngs):
		self.conv = nnx.Conv(3, 64, kernel_size=(7,7), strides=2, padding=3, use_bias=False, rngs=rngs)
		self.bn = nnx.BatchNorm(64, use_running_average=True, rngs=rngs)

	def __call__(self, x:jnp.ndarray):
		x = self.conv(x)
		x = self.bn(x)
		x = nnx.relu(x)

		return nnx.max_pool(x, window_shape=(3,3), strides=(2,2), padding='SAME')

class BlockGroup(nnx.Module):
	def __init__(self, in_channels:int, out_channels:int, num_blocks:int, strides:int, rngs:nnx.Rngs):
		self.layer = nnx.List()
		
		downsample = None
		if strides!=1 or in_channels != out_channels*4:
			downsample = Downsample(in_channels, out_channels*4, strides, rngs=rngs)

		self.layer.append(Bottleneck(in_channels, out_channels, strides, downsample, rngs=rngs))

		for _ in range(1, num_blocks):
			self.layer.append(Bottleneck(out_channels*4, out_channels, strides=1, downsample=None, rngs=rngs))

	def __call__(self, x:jnp.ndarray):
		for layer in self.layer:
			x = layer(x)
		return x

class ResNet(nnx.Module):
	def __init__(self, config:Config, rngs:nnx.Rngs):
		self.stem = Stem(rngs=rngs)

		self.block0 = BlockGroup(64, 64, num_blocks=config.stage_sizes[0], strides=1, rngs=rngs)
		self.block1 = BlockGroup(256, 128, num_blocks=config.stage_sizes[1], strides=2, rngs=rngs)
		self.block2 = BlockGroup(512, 256, num_blocks=config.stage_sizes[2], strides=2, rngs=rngs)
		self.block3 = BlockGroup(1024, 512, num_blocks=config.stage_sizes[3], strides=2, rngs=rngs)

		self.fc = nnx.Linear(2048, config.num_classes, rngs=rngs)

	def __call__(self, x:jnp.ndarray):
		x = self.stem(x)
		x = self.block0(x)
		x = self.block1(x)
		x = self.block2(x)
		x = self.block3(x)

		x = jnp.mean(x, axis=(1,2)) # (B,H,W,C) -> (B,C)
		return self.fc(x)



if __name__=='__main__':
	rngs = nnx.Rngs(42)

	config = Config()
	model = ResNet(config, rngs)

	mock_pixel_values = random.uniform(random.PRNGKey(0), shape =(2,224,224,3), minval = 0.0, maxval=1.0)
	mock_output = model(mock_pixel_values)

	summary = nnx.tabulate(model, mock_pixel_values)
	print(mock_output.shape)
	print(summary)

