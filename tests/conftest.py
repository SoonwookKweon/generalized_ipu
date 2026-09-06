import pandas as pd
import pytest

from generalized_ipu import HierarchyTree


@pytest.fixture
def regions() -> pd.DataFrame:
    return pd.DataFrame({"region_id": ["r1", "r2"], "urban": [1.0, 0.0]})


@pytest.fixture
def households() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "hh_id": ["h1", "h2", "h3", "h4", "h5", "h6"],
            "region_id": ["r1", "r1", "r1", "r2", "r2", "r2"],
            "car": [1.0, 0.0, 2.0, 1.0, 1.0, 0.0],
        }
    )


@pytest.fixture
def persons() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "person_id": [f"p{i}" for i in range(1, 12)],
            "hh_id": ["h1", "h1", "h2", "h3", "h3", "h3", "h4", "h4", "h5", "h6", "h6"],
            "age_group": [
                "child", "adult", "adult", "child", "adult", "adult",
                "adult", "adult", "child", "adult", "adult",
            ],
            "sex": ["M", "F", "M", "F", "M", "F", "M", "F", "M", "F", "M"],
        }
    )


@pytest.fixture
def tree(regions, households, persons) -> HierarchyTree:
    built = HierarchyTree(base_level="household")
    built.add_level("region", regions, primary_key="region_id")
    built.add_level(
        "household", households, primary_key="hh_id", parent_key="region_id", parent_level="region"
    )
    built.add_level(
        "person", persons, primary_key="person_id", parent_key="hh_id", parent_level="household"
    )
    built.validate_integrity()
    return built


@pytest.fixture
def dwellings() -> pd.DataFrame:
    """거처 계층. hh_class 는 그 거처에 속한 가구 수의 계급이다."""
    return pd.DataFrame(
        {
            "dwelling_id": ["d1", "d2", "d3"],
            "hh_class": ["2", "1", "3+"],
        }
    )


@pytest.fixture
def dwelling_households() -> pd.DataFrame:
    """d1 에 두 가구, d2 에 한 가구, d3 에 세 가구가 소속된 다가구 구조."""
    return pd.DataFrame(
        {
            "hh_id": ["h1", "h2", "h3", "h4", "h5", "h6"],
            "dwelling_id": ["d1", "d1", "d2", "d3", "d3", "d3"],
        }
    )


@pytest.fixture
def dwelling_tree(dwellings, dwelling_households, persons) -> HierarchyTree:
    built = HierarchyTree(base_level="household")
    built.add_level("dwelling", dwellings, primary_key="dwelling_id")
    built.add_level(
        "household",
        dwelling_households,
        primary_key="hh_id",
        parent_key="dwelling_id",
        parent_level="dwelling",
    )
    built.add_level(
        "person", persons, primary_key="person_id", parent_key="hh_id", parent_level="household"
    )
    built.validate_integrity()
    return built
