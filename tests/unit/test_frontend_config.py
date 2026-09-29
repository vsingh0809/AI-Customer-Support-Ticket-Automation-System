from uuid import UUID

import pytest

from frontend.streamlit.config import FrontendSettings

DEMO_CUSTOMER_ID = UUID(
    "3f2b8f18-7f3d-4d6e-9a2c-5c3e4a7d8b1f"
)


def test_frontend_settings_loads_valid_configuration() -> None:
    settings = FrontendSettings(
        backend_api_url="http://localhost:8000",
        demo_customer_id=DEMO_CUSTOMER_ID,
    )

    assert settings.backend_api_url == "http://localhost:8000"
    assert settings.demo_customer_id == DEMO_CUSTOMER_ID


def test_frontend_settings_uses_default_backend_url() -> None:
    settings = FrontendSettings(
        demo_customer_id=DEMO_CUSTOMER_ID,
    )

    assert settings.backend_api_url == "http://localhost:8000"


def test_frontend_settings_parses_customer_id_string() -> None:
    settings = FrontendSettings(
        demo_customer_id=(
            "3f2b8f18-7f3d-4d6e-9a2c-5c3e4a7d8b1f"
        ),
    )

    assert settings.demo_customer_id == DEMO_CUSTOMER_ID


def test_frontend_settings_rejects_missing_customer_id() -> None:
    with pytest.raises(ValueError):
        FrontendSettings()


def test_frontend_settings_rejects_invalid_customer_id() -> None:
    with pytest.raises(ValueError):
        FrontendSettings(
            demo_customer_id="not-a-valid-uuid",
        )