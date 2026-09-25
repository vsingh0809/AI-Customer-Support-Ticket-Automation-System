"""Application-level exceptions."""


class ApplicationServiceError(RuntimeError):
    """Raised when an application service cannot complete an operation."""