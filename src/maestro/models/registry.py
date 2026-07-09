from .transformers import TransformersModel


BACKENDS = {
    "transformers": TransformersModel,
}


def get_model(
    model_name,
    backend="transformers",
    **kwargs,
):
    if backend not in BACKENDS:
        available = ", ".join(BACKENDS)

        raise ValueError(
            f"Unknown backend: {backend}. "
            f"Available backends: {available}"
        )

    model_cls = BACKENDS[backend]

    return model_cls(
        model_name=model_name,
        **kwargs,
    )
