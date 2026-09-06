"""generalized-ipu: 다계층, 구간 목표, 다차원 교차표, 범주 위계를 수용하는 IPU 라이브러리.

표본 미시 자료의 각 개체에 가중치를 부여하여, 가중합이 외부에서 주어진 다수의
집계 통계를 동시에 재현하도록 조정한다. 용어와 기호의 정의는 `docs/01_용어와_정의.md`
를 따르며, 이 모듈은 각 하위 패키지의 공개 이름을 한 곳에 모아 노출한다.
"""

from .constraints import (
    AbstractConstraint,
    BoundChecker,
    ConstraintRegistry,
    IntervalTarget,
    LinearBalance,
    LogicalRuleParser,
    MissingMarginEstimator,
    StructuralZeroMask,
    TensorTarget,
    ZeroCellResolver,
    balance_from_classes,
    marginal_from_series,
)
from .diagnostics import (
    ConvergenceTracker,
    FitReportGenerator,
    WeightDistributionAnalyzer,
)
from .engine import GeneralizedIPUEngine, IPUResult
from .hierarchy import AggregationMapper, CategoryHierarchy, HierarchyTree, MappingMatrix
from .io import DataLoader, DatasetExporter
from .matrix import (
    AttributeMatrix,
    CoarseBlock,
    ColumnBlock,
    CrossTabBlock,
    ShareBlock,
    UnifiedSparseMatrixBuilder,
)

__version__ = "0.1.0.dev0"

__all__ = [
    "AbstractConstraint",
    "AggregationMapper",
    "AttributeMatrix",
    "BoundChecker",
    "CategoryHierarchy",
    "CoarseBlock",
    "ColumnBlock",
    "ConstraintRegistry",
    "ConvergenceTracker",
    "CrossTabBlock",
    "DataLoader",
    "DatasetExporter",
    "FitReportGenerator",
    "GeneralizedIPUEngine",
    "HierarchyTree",
    "IPUResult",
    "IntervalTarget",
    "LinearBalance",
    "LogicalRuleParser",
    "MappingMatrix",
    "MissingMarginEstimator",
    "ShareBlock",
    "StructuralZeroMask",
    "TensorTarget",
    "UnifiedSparseMatrixBuilder",
    "WeightDistributionAnalyzer",
    "ZeroCellResolver",
    "balance_from_classes",
    "marginal_from_series",
    "__version__",
]
