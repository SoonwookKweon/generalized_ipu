"""제약 조건의 정의, 목표 구간 정규화, 영 셀과 결측 셀의 처리."""

from .base import AbstractConstraint
from .interval import (
    BoundChecker,
    IntervalTarget,
    bound_violations,
    compute_update_ratios,
    normalize_targets,
)
from .missing import (
    LinearBalance,
    MissingMarginEstimator,
    balance_from_classes,
    marginal_from_series,
)
from .registry import ConstraintRegistry
from .rules import LogicalRuleParser
from .tensor import TensorFlattener, TensorTarget, interval_tensor_target
from .zeros import StructuralZeroMask, ZeroCellResolver

__all__ = [
    "AbstractConstraint",
    "BoundChecker",
    "IntervalTarget",
    "bound_violations",
    "compute_update_ratios",
    "normalize_targets",
    "ConstraintRegistry",
    "LinearBalance",
    "LogicalRuleParser",
    "MissingMarginEstimator",
    "TensorFlattener",
    "TensorTarget",
    "balance_from_classes",
    "interval_tensor_target",
    "marginal_from_series",
    "StructuralZeroMask",
    "ZeroCellResolver",
]
