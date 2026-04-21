
import torch


def get_cuda_info() -> dict[str, Any]:
    """
    Get information about the CUDA devices available in the system.
    author : Timothy Gebhard
    """

    # No CUDA devices available
    if not torch.cuda.is_available():
        return {}

    # CUDA devices are available
    return {
        "cuDNN version": torch.backends.cudnn.version(),  # type: ignore
        "CUDA version": torch.version.cuda,
        "device count": torch.cuda.device_count(),
        "device name": torch.cuda.get_device_name(0),
        "memory (GB)": round(
            torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1
        ),
    }