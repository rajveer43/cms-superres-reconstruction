from .swinir import SwinIRConfig

MODEL_REGISTRY = {
    "swinir_base": SwinIRConfig,
    "swinir_film": SwinIRConfig,  # toggle use_film=True via config
}
