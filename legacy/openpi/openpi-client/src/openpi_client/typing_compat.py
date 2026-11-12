try:
    from typing import override
except ImportError:  # pragma: no cover - Python < 3.12
    try:
        from typing_extensions import override
    except ImportError:  # pragma: no cover
        def override(func):
            return func
