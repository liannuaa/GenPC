import numpy as np
import pytest

from src.prior_backend_contract import (
    NativePriorContract, apply_native_to_method, validate_proper_sim3,
)


def test_backend_contract_accepts_proper_isotropic_mapping():
    transform = np.eye(4)
    transform[:3, :3] *= 2.
    transform[:3, 3] = (1., 2., 3.)
    contract = NativePriorContract("test", transform, "native")
    assert validate_proper_sim3(transform) == pytest.approx(2.)
    assert np.allclose(apply_native_to_method(np.array([[1., 0., 0.]]), contract), [[3., 2., 3.]])


def test_backend_contract_rejects_backend_specific_anisotropic_hack():
    transform = np.eye(4)
    transform[0, 0] = 2.
    with pytest.raises(ValueError, match="anisotropic"):
        validate_proper_sim3(transform)
