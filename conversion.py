import jax
from flax import serialization, traverse_util, nnx
from modeling import Config, ResNet

def load_flax_weights(model, msgpack_path="flax_model.msgpack"):
	with open(msgpack_path, "rb") as f:
		raw_bytes = f.read()

	msgpack_params = serialization.msgpack_restore(raw_bytes)
	flat_msgpack_params = traverse_util.flatten_dict(msgpack_params, keep_empty_nodes=False)

	model_state = nnx.state(model)
	flat_model_state = nnx.to_flat_state(model_state)

	nnx_dict = {".".join(str(x) for x in path): var for path, var in flat_model_state}
	aligned_params = {}
	matched_keys = set()

	for hf_tuple_path, array in flat_msgpack_params.items():
		hf_path = ".".join(str(x) for x in hf_tuple_path)
		my_path = hf_path

		if "batch_stats.resnet.encoder." in my_path:
			my_path = my_path.replace("batch_stats.resnet.encoder.","")
		if "params.resnet.encoder." in my_path:
			my_path = my_path.replace("params.resnet.encoder.","")
		if "stages." in my_path:
			my_path = my_path.replace("stages.","block")
		if "layers" in my_path:
			my_path = my_path.replace("layers","layer")

		for i in range(6):
			if f"layer.{i}.normalization" in my_path:
				my_path = my_path.replace(f"layer.{i}.normalization",f"bn{i}")
			if f"layer.{i}.convolution.kernel" in my_path:
				my_path = my_path.replace(f"layer.{i}.convolution.kernel",f"conv{i}.kernel")
		
		if "shortcut" in my_path:
			my_path = my_path.replace("shortcut","downsample")
		if "downsample.convolution" in my_path:
				my_path = my_path.replace("convolution","conv")
		if "downsample.normalization" in my_path:
				my_path = my_path.replace("normalization","bn")
		if "params.classifier.1" in my_path:
				my_path = my_path.replace("params.classifier.1","fc")
		if "embedder.normalization" in my_path:
				my_path = "stem.bn" + my_path.rpartition(".embedder.normalization")[2]
		if "embedder.convolution" in my_path:
				my_path = "stem.conv.kernel"

		if my_path in nnx_dict:
			expected_shape = nnx_dict[my_path].get_value().shape
			if array.shape == expected_shape:
				tuple_key = tuple(int(x) if x.isdigit() else x for x in my_path.split("."))
				aligned_params[tuple_key] = nnx.Variable(array)
				matched_keys.add(my_path)
			else:
				print(f"Skipped Mismatch on {my_path}: Got {array.shape}, expected {expected_shape}")
		else:
			print(f"Skipped / Unmapped HF track: {hf_path} -> Attempted as: {my_path}")

	all_model_keys = set(nnx_dict.keys())
	uninitialized_keys = all_model_keys - matched_keys

	if uninitialized_keys:
		print(f"WARNING: {len(uninitialized_keys)} parameters in model were not loaded.")
		print("\n"+"="*20 + " UNINITIALIZED PARAMS " + "="*20)
		for i in uninitialized_keys:
			print(i)
	new_state = nnx.State.from_flat_path(aligned_params)
	nnx.update(model, new_state)
	
	return model