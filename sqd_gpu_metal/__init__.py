"""Native MLX/Metal Selected Basis Diagonalization for Apple Silicon."""

from .rdm import check_mixed_rdm_size, mixed_rdm_q60
from .solver import (
    BackendUnavailableError,
    MetalDavidsonResult,
    MetalSelectedCIContractor,
    MetalSelectedS2Contractor,
    absorb_h1e,
    davidson_lowest,
    make_hdiag,
    make_spin_square_diagonal,
    metal_gpu_execution,
)

__version__ = "1.5.0"

__all__ = [
    "BackendUnavailableError",
    "MetalDavidsonResult",
    "MetalSelectedCIContractor",
    "MetalSelectedS2Contractor",
    "absorb_h1e",
    "check_mixed_rdm_size",
    "davidson_lowest",
    "make_hdiag",
    "make_spin_square_diagonal",
    "metal_gpu_execution",
    "mixed_rdm_q60",
]
