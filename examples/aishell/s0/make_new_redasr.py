old_model = "/data2/wangyuwen/output/wenet_firered.pt"
new_model = "/data2/wangyuwen/output/new_wenet_firered.pt"

import torch

model_dict = torch.load(old_model)
new_mode_dict = {}

for module_name in model_dict.keys():
    if "decoder.embed" in module_name:
        print(module_name)
        new_mode_dict[module_name] = model_dict[module_name]
    elif "decoder.decoders.0" in module_name:
        new_mode_dict[module_name] = model_dict[module_name]
    elif "decoder.decoders.15" in module_name:
        new_module_name = module_name.replace("decoder.decoders.15", "decoder.decoders.1")
        new_mode_dict[new_module_name] = model_dict[module_name]
    elif "decoder.output_layer"in module_name or "decoder.after_norm" in module_name:
        new_mode_dict[module_name] = model_dict[module_name]
        print(module_name)
torch.save(new_mode_dict, new_model)