import numpy as np
import pandas as pd
import pytest

from generalized_ipu import CategoryHierarchy, HierarchyTree, MappingMatrix
from generalized_ipu.hierarchy.mapper import AggregationMapper


def test_relation_to_base(tree):
    assert tree.relation_to_base("household") == "base"
    assert tree.relation_to_base("person") == "descendant"
    assert tree.relation_to_base("region") == "ancestor"


def test_duplicate_primary_key_is_rejected(households):
    duplicated = pd.concat([households, households.iloc[[0]]], ignore_index=True)
    built = HierarchyTree("household")
    built.add_level("household", duplicated, primary_key="hh_id")
    with pytest.raises(ValueError, match="중복"):
        built.validate_integrity()


def test_referential_integrity_is_checked(households, persons, regions):
    broken = persons.copy()
    broken.loc[0, "hh_id"] = "h99"
    built = HierarchyTree("household")
    built.add_level("region", regions, primary_key="region_id")
    built.add_level(
        "household", households, primary_key="hh_id", parent_key="region_id", parent_level="region"
    )
    built.add_level(
        "person", broken, primary_key="person_id", parent_key="hh_id", parent_level="household"
    )
    with pytest.raises(ValueError, match="대응하지 않"):
        built.validate_integrity()


def test_descendant_aggregation_counts_children(tree, persons):
    persons_with_flag = tree.nodes["person"].data.copy()
    persons_with_flag["is_adult"] = (persons_with_flag["age_group"] == "adult").astype(float)
    tree.nodes["person"].data = persons_with_flag

    aggregated = AggregationMapper.aggregate_to_base(tree, "person", ["is_adult"])
    assert list(aggregated["is_adult"]) == [1.0, 1.0, 2.0, 2.0, 0.0, 2.0]


def test_ancestor_broadcast_repeats_parent_attribute(tree):
    aggregated = AggregationMapper.aggregate_to_base(tree, "region", ["urban"])
    assert list(aggregated["urban"]) == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]


def test_base_level_aggregation_is_identity(tree):
    aggregated = AggregationMapper.aggregate_to_base(tree, "household", ["car"])
    assert list(aggregated["car"]) == [1.0, 0.0, 2.0, 1.0, 1.0, 0.0]


def test_crosstab_follows_c_order(tree):
    dims = {"age_group": ["child", "adult"], "sex": ["M", "F"]}
    block, names = AggregationMapper.crosstab_to_base(tree, "person", dims)

    assert names == [
        "age_group=child|sex=M",
        "age_group=child|sex=F",
        "age_group=adult|sex=M",
        "age_group=adult|sex=F",
    ]
    assert block.shape == (6, 4)
    # 전체 인원수가 보존된다.
    assert block.sum() == 11.0
    # h1: child M 1명, adult F 1명
    assert list(np.asarray(block[0].todense()).ravel()) == [1.0, 0.0, 0.0, 1.0]


def test_mapping_matrix_rejects_uncovered_fine_category():
    hierarchy = CategoryHierarchy("age_sex")
    hierarchy.add_mapping("child", ["age_group=child|sex=M"])
    with pytest.raises(ValueError, match="속하지 않은"):
        MappingMatrix.build_binary_matrix(
            hierarchy, ["age_group=child|sex=M", "age_group=adult|sex=M"]
        )


def test_mapping_matrix_rejects_duplicate_fine_category():
    hierarchy = CategoryHierarchy("age_sex")
    hierarchy.add_mapping("a", ["f1"])
    with pytest.raises(ValueError, match="중복 포섭"):
        hierarchy.add_mapping("b", ["f1"])


def test_mapping_matrix_shape_and_values():
    hierarchy = CategoryHierarchy("age_sex")
    hierarchy.add_mapping("child", ["c_m", "c_f"])
    hierarchy.add_mapping("adult", ["a_m", "a_f"])
    matrix = MappingMatrix.build_binary_matrix(hierarchy, ["c_m", "c_f", "a_m", "a_f"])

    assert matrix.shape == (2, 4)
    assert np.array_equal(
        matrix.toarray(), np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
    )
