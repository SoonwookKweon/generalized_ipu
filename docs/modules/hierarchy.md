# `generalized_ipu.hierarchy` — 계층 및 범주 위계 관리

계층 트리와 범주 위계를 정의하고 검증하는 패키지이다. 다른 내부 패키지에 의존하지 않으며, 외부 의존은 `pandas`, `numpy`, `scipy.sparse`에 한정된다.

**책임 범위**

- 계층별 원본 자료를 주키와 부모키로 연결하고 무결성을 검증한다.
- 기본 계층을 지정하고, 다른 계층의 속성을 기본 계층 좌표계로 집계한다.
- 상위 계층 개체의 수를 분수 귀속으로 기본 계층 좌표계에서 계상한다.
- 소범주와 대범주의 포섭 관계를 이진 매핑 행렬로 변환한다.

**책임 밖**

- 속성 행렬 자체의 구축은 `matrix` 패키지가 담당한다.
- 목표값의 해석은 `constraints` 패키지가 담당한다.

---

## `tree.py` — 계층 트리

### `HierarchyNode`

하나의 계층을 표현하는 자료 보관 객체이다.

| 속성 | 형 | 의미 |
| --- | --- | --- |
| `name` | `str` | 계층명 |
| `data` | `pd.DataFrame` | 해당 계층의 원본 표 |
| `primary_key` | `str` | 주키 열 이름 |
| `parent_key` | `Optional[str]` | 부모키 열 이름. 최상위 계층은 `None` |
| `parent_level` | `Optional[str]` | 부모 계층명. 생략하면 부모키로부터 추론한다 |

생성자의 `copy` 인자가 참이면 자료의 복사본을 보관하므로 호출자가 원본을 변경해도 노드의 상태가 유지된다. 자료가 큰 경우 `copy=False`로 사본 생성을 생략할 수 있다.

### `HierarchyTree`

계층들의 집합과 그 부모-자식 관계를 관리한다.

| 메서드 | 설명 |
| --- | --- |
| `add_level(level_name, data, primary_key, parent_key=None, parent_level=None, copy=True)` | 계층을 등록한다. 같은 이름을 다시 등록하면 예외가 발생한다 |
| `base_node` | 기본 계층 노드를 반환하는 속성 |
| `resolve_parent_level(level_name)` | 부모 계층명을 반환한다. 명시되지 않았으면 부모키를 주키로 가지는 계층에서 추론하며, 후보가 없거나 여럿이면 예외를 발생시킨다 |
| `ancestors(level_name)` | 근(root)까지의 상위 계층을 가까운 순으로 반환한다. 순환을 발견하면 예외를 발생시킨다 |
| `path_from_base(level_name)` | 기본 계층에서 대상 계층까지 내려가는 경로를 반환한다. 하위 계층이 아니면 빈 목록이다 |
| `relation_to_base(level_name)` | `"base"`, `"descendant"`, `"ancestor"` 중 하나를 반환한다. 어느 쪽으로도 연결되지 않으면 예외를 발생시킨다 |
| `validate_integrity()` | 무결성 조건 네 가지를 모두 검사한다 |

**검사 항목**

1. 기본 계층이 등록되어 있는가. 위반 시 `ValueError`.
2. 각 노드의 부모키가 그 노드의 자료에 열로 존재하는가. 위반 시 `KeyError`.
3. 부모키 값이 상위 계층 주키 값의 부분집합인가. 결측 부모키도 위반으로 본다. 위반 시 `ValueError`.
4. 주키에 중복 값이 없는가. 위반 시 `ValueError`.

조건 3이 검증되지 않으면 결합에 실패한 행이 집계 단계에서 0으로 채워져 자료 오류가 조용한 과소 집계로 나타난다. 조건 4가 검증되지 않으면 결합 과정에서 행이 증식하여 기본 단위의 수가 달라진다.

---

## `category.py` — 범주 위계

### `CategoryHierarchy`

한 속성에 대한 소범주와 대범주의 포섭 관계를 등록한다.

| 메서드 및 속성 | 설명 |
| --- | --- |
| `name` | 위계의 이름. 대상 속성명을 사용한다 |
| `mapping: Dict[str, List[str]]` | 대범주명에서 소범주명 목록으로의 대응 |
| `fine_categories: List[str]` | 등록 순서대로 누적된 소범주 목록 |
| `coarse_categories: List[str]` | 대범주 등록 순서 |
| `add_mapping(coarse, fines)` | 대범주 하나와 그에 속한 소범주 목록을 등록한다. 대범주명이 중복되거나 소범주가 둘 이상의 대범주에 포섭되면 예외를 발생시킨다 |
| `validate(fine_categories=None)` | 원자 스키마 전체와 대조하여 누락 소범주와 미등록 소범주를 걸러낸다 |

`fine_categories`의 순서는 매핑 행렬의 열 순서이자 소범주 가중합 벡터 $\mathbf{S}_{fine}$의 성분 순서이므로, 속성 행렬의 제약 열 순서와 명시적으로 동기화되어야 한다. `MappingMatrix`에 소범주 목록을 직접 넘기면 이 동기화를 강제할 수 있다.

### `MappingMatrix`

`build_binary_matrix(hierarchy, fine_categories=None) -> csr_matrix`는 등록된 위계로부터 $M \in \{0,1\}^{K_c \times K_f}$를 생성한다. 행은 대범주 등록 순서, 열은 `fine_categories`의 순서를 따르며, 생략하면 위계에 등록된 소범주 순서를 사용한다. 좌표 형식으로 직접 구축하므로 조밀 중간 산출물이 생기지 않는다.

생성 직후 각 열의 합이 1인지 검사하여, 어느 대범주에도 속하지 않은 소범주나 중복 포섭을 예외로 걸러낸다.

**용도.** 대범주 목표만 주어진 제약을 소범주 좌표계에서 다루기 위한 변환이다. 대범주 가중합은 $\mathbf{S}_{coarse} = M\,\mathbf{S}_{fine}$이며, 속성 행렬 수준에서는 $A_{coarse} = A_{fine} M^{\top}$으로 계산한다.

---

## `mapper.py` — 기본 단위 선정과 집계

### `BaseUnitSelector`

계층 트리의 무결성을 검증한 뒤 기본 계층 노드를 확정한다. `n_units`는 기본 단위의 수, `keys`는 기본 단위의 주키를 행 순서대로 담은 `Series`이다. 이 행 순서가 속성 행렬의 행 순서이자 가중치 벡터의 성분 순서이다.

### `AggregationMapper`

#### `aggregate_to_base(tree, target_level, attribute_cols) -> pd.DataFrame`

대상 계층의 속성 열을 기본 계층의 행 좌표계로 옮긴다. 기본 계층과의 관계에 따라 세 경로로 갈린다.

| 관계 | 처리 |
| --- | --- |
| `base` | 항등 사상. 지정한 속성 열을 그대로 반환한다 |
| `descendant` | 경로를 따라 부모키 기준 합산을 반복하여 기본 계층까지 올린다. 소속된 행이 없는 기본 단위는 0이다 |
| `ancestor` | 상위 계층 행의 속성 값을 그에 속한 모든 기본 단위에 방송한다 |

반환 표의 행 수와 순서는 기본 계층 자료와 같고, 열은 `attribute_cols`이며 자료형은 `float64`이다. 두 계층 이상 떨어진 하위 계층도 중간 계층을 거쳐 누적 합산한다.

#### `crosstab_to_base(tree, target_level, dims, weight_col=None) -> Tuple[csr_matrix, List[str]]`

대상 계층의 $N$개 변수 결합 분포를 기본 단위별 셀 수량으로 집계한다.

- `dims`는 차원 이름에서 범주 목록으로의 대응이며, 선언 순서가 축 순서이다.
- 열 순서는 C 순서 평탄화(마지막 차원이 가장 빠르게 변함)를 따른다.
- `weight_col`을 주면 행마다 1 대신 그 열의 값을 더한다.
- 어느 차원에서든 범주 목록에 없는 값을 가진 행은 기여량이 0이다.

좌표 형식으로 직접 구축하므로 $n \times K$ 조밀 배열을 만들지 않는다. 반환되는 열 이름은 `age_group=adult|sex=F` 형식이다.

#### `descendant_count_to_base(tree, target_level) -> pd.Series`

기본 단위별로 소속된 하위 계층 개체의 수를 반환한다. 기본 계층이 가구이고 대상 계층이 가구원이면 이 값이 가구 규모, 곧 $n$인가구의 $n$이다. 소속 개체가 없는 기본 단위는 0이고, 대상이 기본 계층 자신이면 모두 1이다.

#### `size_class_labels(counts, boundaries, labels=None) -> pd.Series`

개체 수를 계급 이름으로 변환한다. `boundaries`는 오름차순 하한의 나열이며 마지막 계급은 상한을 두지 않는다. `boundaries=(1, 2, 3)`은 `'1'`, `'2'`, `'3+'` 세 계급을 뜻한다. 최소 하한 미만의 값이 있으면 계급 정의가 자료를 덮지 못한다는 뜻이므로 예외가 발생한다.

`descendant_count_to_base`와 함께 쓰면 $n$인가구 계급 열을 얻는다. 이 열을 기본 계층 자료에 붙이고 `CrossTabBlock`으로 구축하면 가구 규모별 가구 수가 제약 열이 된다.

#### `ancestor_key_of_base(tree, target_level) -> pd.Series`

각 기본 단위가 소속된 상위 계층 개체의 주키를 기본 단위 행 순서로 반환한다. 두 계층 이상 떨어진 상위 계층도 중간 계층의 부모키를 따라 거슬러 올라간다.

#### `ancestor_share_to_base(tree, target_level, dims=None) -> Tuple[csr_matrix, List[str]]`

상위 계층 개체를 기본 단위 좌표계에서 **한 번만** 세는 분수 귀속 행렬을 만든다. 상위 계층 개체 하나에 기본 단위 $m$개가 소속되면 각 기본 단위에 $1/m$을 부여한다.

방송과 구별해야 한다. `aggregate_to_base`의 `ancestor` 경로는 상위 계층의 값을 소속 기본 단위마다 복제하므로, 값이 1인 열을 방송하면 가중합이 상위 계층 개체 수가 아니라 기본 단위 수가 된다. 거처 계층에 가구 계층이 매달린 다가구주택 구조에서 "거처 수"를 제약하려면 반드시 이 함수를 써야 한다.

| 상황 | 사용할 함수 | 가중합의 의미 |
| --- | --- | --- |
| 상위 계층의 수량 속성(면적, 소득 등) | `aggregate_to_base` | 기본 단위별로 복제된 값의 합 |
| 상위 계층의 개체 수 | `ancestor_share_to_base` | 상위 계층 개체 수 |

`dims`를 주면 상위 계층의 변수 조합별로 열을 나누고, 생략하면 개체 수 전체를 세는 한 개의 열(`계층명.count`)을 만든다.

**전제.** 한 상위 개체에 속한 기본 단위들의 가중치가 서로 같을 때 해당 열의 가중합이 상위 계층 개체 수와 정확히 일치한다. 가중치가 벌어지면 가중치의 평균으로 계상되므로, 상위 계층 제약은 이 전제 위에서 해석해야 한다.

#### `cell_names(dim_names, categories) -> List[str]`

C 순서 평탄화에 대응하는 셀 이름을 생성한다. `constraints.tensor.TensorFlattener.cell_names`와 동일한 규칙을 사용하므로, 목표 텐서의 열 이름과 속성 행렬의 열 이름이 일치한다.
