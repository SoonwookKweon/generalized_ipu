import numpy as np
import pytest

from generalized_ipu import (
    BoundChecker,
    CategoryHierarchy,
    CoarseBlock,
    ColumnBlock,
    ConstraintRegistry,
    CrossTabBlock,
    DatasetExporter,
    FitReportGenerator,
    GeneralizedIPUEngine,
    LogicalRuleParser,
    TensorTarget,
    UnifiedSparseMatrixBuilder,
    WeightDistributionAnalyzer,
)

DIMS = {"age_group": ["child", "adult"], "sex": ["M", "F"]}


def build_attribute_matrix(tree):
    builder = UnifiedSparseMatrixBuilder(tree)
    return builder.build(
        [
            ColumnBlock(name="hh_car", level="household", cols=["car"]),
            CrossTabBlock(name="person_age_sex", level="person", dims=DIMS),
        ]
    )


def test_builder_reports_block_slices_and_column_names(tree):
    attribute = build_attribute_matrix(tree)

    assert attribute.n_units == 6
    assert attribute.n_columns == 5
    assert attribute.block_slices["hh_car"] == slice(0, 1)
    assert attribute.block_slices["person_age_sex"] == slice(1, 5)
    assert attribute.column_names[0] == "household.car"
    assert attribute.column_names[1] == "age_group=child|sex=M"
    assert list(attribute.base_keys) == ["h1", "h2", "h3", "h4", "h5", "h6"]


def test_builder_drops_structural_zero_columns_at_build_time(tree):
    dims = {"age_group": ["child", "adult"], "license": [0, 1]}
    persons = tree.nodes["person"].data.copy()
    persons["license"] = [0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1]
    tree.nodes["person"].data = persons

    constraint = TensorTarget("t", "person", dims, np.ones((2, 2)))
    parser = LogicalRuleParser(["age_group == 'child' and license == 1"])
    mask = parser.build_validity_mask(constraint.category_frame())

    builder = UnifiedSparseMatrixBuilder(tree)
    attribute = builder.build(
        [CrossTabBlock(name="t", level="person", dims=dims)],
        validity_masks={"t": mask},
    )

    assert attribute.n_columns == 3
    assert "age_group=child|license=1" not in attribute.column_names


def test_coarse_block_aggregates_fine_columns(tree):
    hierarchy = CategoryHierarchy("age")
    hierarchy.add_mapping("child", ["age_group=child|sex=M", "age_group=child|sex=F"])
    hierarchy.add_mapping("adult", ["age_group=adult|sex=M", "age_group=adult|sex=F"])

    builder = UnifiedSparseMatrixBuilder(tree)
    attribute = builder.build(
        [
            CrossTabBlock(name="fine", level="person", dims=DIMS),
            CoarseBlock(name="coarse", source="fine", hierarchy=hierarchy),
        ]
    )

    fine = attribute.block("fine").toarray()
    coarse = attribute.block("coarse").toarray()

    assert coarse.shape == (6, 2)
    assert np.allclose(coarse[:, 0], fine[:, 0] + fine[:, 1])
    assert np.allclose(coarse[:, 1], fine[:, 2] + fine[:, 3])


def test_registry_and_matrix_agree_on_column_count(tree):
    attribute = build_attribute_matrix(tree)
    registry = ConstraintRegistry()
    registry.register(BoundChecker("hh_car", "household", {"car": 400.0}))
    registry.register(
        TensorTarget("person_age_sex", "person", DIMS, np.array([[150.0, 150.0], [400.0, 400.0]]))
    )

    assert registry.validate_against(attribute.n_columns)


def test_end_to_end_pipeline_converges(tree):
    attribute = build_attribute_matrix(tree)
    registry = ConstraintRegistry()
    registry.register(BoundChecker("hh_car", "household", {"car": (350.0, 450.0)}))
    registry.register(
        TensorTarget("person_age_sex", "person", DIMS, np.array([[150.0, 150.0], [400.0, 400.0]]))
    )

    engine = GeneralizedIPUEngine.from_registry(
        attribute.matrix, registry, relative_gap=0.02, max_iterations=300
    )
    result = engine.fit()

    assert result.converged, result.report
    assert np.all(result.weights > 0)

    report = FitReportGenerator.generate_report(
        result.weighted_sums,
        *registry.bounds(),
        column_names=attribute.column_names,
        column_owners=registry.column_owners,
        relative_gap=0.02,
    )
    assert report["relative_gap"].max() <= 0.02
    assert report.iloc[0]["relative_gap"] >= report.iloc[-1]["relative_gap"]
    # 보고서의 판정이 엔진의 수렴 판정과 일치한다.
    assert bool(report["satisfied"].all())

    summary = FitReportGenerator.summarize_by_constraint(report)
    assert set(summary["constraint"]) == {"hh_car", "person_age_sex"}
    assert summary["max_relative_gap"].max() <= 0.02
    assert int(summary["n_unsatisfied"].sum()) == 0


def test_report_tolerance_separates_strict_and_admitted_columns():
    report = FitReportGenerator.generate_report(
        np.array([100.0, 100.0]),
        np.array([101.0, 200.0]),
        column_names=["close", "far"],
        column_owners=["c", "c"],
        relative_gap=0.05,
    )
    strict = report.set_index("column")["within_bounds"]
    admitted = report.set_index("column")["satisfied"]

    assert not strict["close"] and not strict["far"]
    assert admitted["close"] and not admitted["far"]

    summary = FitReportGenerator.summarize_by_constraint(report)
    assert int(summary["n_outside_bounds"].iloc[0]) == 2
    assert int(summary["n_unsatisfied"].iloc[0]) == 1


def test_weight_distribution_report(tree):
    attribute = build_attribute_matrix(tree)
    engine = GeneralizedIPUEngine(
        attribute.matrix,
        lower=np.array([400.0, 150.0, 150.0, 400.0, 400.0]),
        relative_gap=0.05,
        max_iterations=300,
    )
    result = engine.fit()

    report = WeightDistributionAnalyzer.analyze(
        result.weights, initial_weights=np.ones(attribute.n_units), weight_floor=1e-5
    )
    assert report.n_units == 6
    assert report.effective_sample_size <= report.n_units
    assert 0.0 <= report.normalized_entropy <= 1.0
    assert report.top_5_percent_share >= 1.0 / report.n_units
    assert isinstance(WeightDistributionAnalyzer.flag_concerns(report), list)


def test_exporter_joins_on_primary_key(tree):
    attribute = build_attribute_matrix(tree)
    weights = np.arange(1.0, 7.0)

    attached = DatasetExporter.attach_weights(
        tree.nodes["household"].data,
        weights,
        keys=attribute.base_keys,
        key_column="hh_id",
    )
    assert list(attached["ipu_weight"]) == list(weights)


def test_exporter_propagates_weights_to_person_level(tree):
    weights = np.arange(1.0, 7.0)
    propagated = DatasetExporter.propagate_to_level(tree, "person", weights)

    assert len(propagated) == 11
    assert propagated.loc[propagated["hh_id"] == "h3", "ipu_weight"].tolist() == [3.0, 3.0, 3.0]


def test_integerization_preserves_expected_total():
    weights = np.full(1000, 2.5)
    integerized = DatasetExporter.integerize(weights, random_state=0)

    assert integerized.dtype == np.int64
    assert integerized.sum() == pytest.approx(2500, rel=0.05)
