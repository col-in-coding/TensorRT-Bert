import ctypes
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

handle = ctypes.CDLL("/workspace/Github/TensorRT-Bert/LayerNormPlugin/LayerNorm.so", mode=ctypes.RTLD_GLOBAL)
if not handle:
    raise RuntimeError("Could not load plugin library. Is `LayerNorm.so` on your LD_LIBRARY_PATH?")

trt.init_libnvinfer_plugins(TRT_LOGGER, "")

if __name__ == "__main__":
    plg_registry = trt.get_plugin_registry()
    for c in plg_registry.plugin_creator_list:
        print("===> ", c.name, c.plugin_version)