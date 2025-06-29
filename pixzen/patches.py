# patches.py
import sys
import types
import importlib.machinery

def apply_torchvision_patches():
    try:
        import torchvision.transforms
        from torchvision.transforms import functional as TF_functional_module
    except ImportError:
        print("PixZen Patch ERROR: Failed to import torchvision. Cannot apply patches.")
        return
    
    expected_module_fqn = "torchvision.transforms.functional_tensor"
    submodule_name_to_create = "functional_tensor"
    
    tv_transforms_module = torchvision.transforms

    if hasattr(tv_transforms_module, submodule_name_to_create):
         dummy_module = getattr(tv_transforms_module, submodule_name_to_create)
    else:
        dummy_module = types.ModuleType(expected_module_fqn)
        setattr(tv_transforms_module, submodule_name_to_create, dummy_module)

    if expected_module_fqn not in sys.modules:
        sys.modules[expected_module_fqn] = dummy_module
    
    if not hasattr(dummy_module, "rgb_to_grayscale"):
        dummy_module.rgb_to_grayscale = TF_functional_module.rgb_to_grayscale