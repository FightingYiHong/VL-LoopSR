from .api_backends.openai_compatible_backend import OpenAICompatibleBackend
from .api_backends.custom_completion_backend import CustomCompletionBackend


def get_backend(config):
    backend_type = config.backend_type.lower()

    if backend_type == "openai_compatible":
        return OpenAICompatibleBackend(config)

    if backend_type == "custom_completion":
        return CustomCompletionBackend(config)

    if backend_type == "transformers":
        from .local_backends.transformers_backend import TransformersBackend

        return TransformersBackend(config)

    raise ValueError(f"不支持的 backend_type: {config.backend_type}")
