# `generalized_ipu.constraints` — 제약 조건 정의, 영 셀과 결측 셀 처리

목표값을 객체화하고, 갱신 비율의 산출 규칙을 정의하며, 구조적 영과 표본 영을 처리하는 패키지이다. 다른 내부 패키지에 의존하지 않는다.

**책임 범위**

- 스칼라 목표와 구간 목표를 하나의 표현으로 정규화한다.
- $N$차원 교차표를 제약 열의 나열로 평탄화한다.
- 도메인 규칙으로부터 제약 열의 유효 마스크를 산출한다.
- 표본 영에 의사 빈도를 주입한다.
- 주변표의 결측 셀에 대한 목표 구간을 균형식으로부터 연역한다.

**책임 밖**

- 가중치의 갱신은 `matrix`와 `engine`이 담당한다. 본 패키지는 비율 $r_k$까지만 산출한다.

---

## `base.py` — 제약 추상 인터페이스

### `AbstractConstraint`

모든 제약 객체가 상속하는 기반 클래스이다. 제약은 이름, 대상 계층, 제약 열 이름, 그리고 열별 목표 구간으로 구성된다.

| 요소 | 설명 |
| --- | --- |
| `__init__(name, level, column_names, lower, upper)` | 목표를 `float64` 배열로 보관한다. 길이 불일치, 음수 하한, 하한이 상한보다 큰 경우를 생성 시점에 예외로 걸러낸다 |
| `n_columns` | 이 제약이 점유하는 제약 열의 수 |
| `compute_ratio(weighted_sums)` | 구간별 갱신 비율을 반환한다. 기정 구현은 `interval.compute_update_ratios`를 위임 호출한다 |
| `bound_violations(weighted_sums)` | 제약 열별 구간 위반량 $v_k$를 반환한다 |

**계약.** 입력과 출력의 길이는 `n_columns`와 같고, 반환 자료형은 `float64`이며, 조정이 불필요한 제약 열에는 정확히 1.0이 들어간다.

`level` 속성은 이 제약이 어느 계층의 집계량을 대상으로 하는지를 나타내며, 속성 행렬 구축 시 어떤 집계 사상을 적용할지 결정하는 데 쓰인다.

---

## `interval.py` — 구간 목표

### `IntervalTarget`

`lower`와 `upper`를 가지는 동결 데이터 클래스이다. `from_value`는 실수, 두 값 순서쌍, 또는 `IntervalTarget` 자신을 받아 구간으로 정규화한다. 스칼라 목표는 `lower == upper`인 퇴화 구간이 되며 `is_scalar`가 참이다.

### 모듈 함수

| 함수 | 설명 |
| --- | --- |
| `normalize_targets(targets)` | 목표 나열을 `(lower, upper)` 배열 쌍으로 정규화한다 |
| `compute_update_ratios(weighted_sums, lower, upper)` | 구간별 갱신 비율을 벡터 연산으로 산출한다 |
| `bound_violations(weighted_sums, lower, upper)` | $v_k = \max(L_k - S_k,\ S_k - U_k,\ 0)$을 산출한다 |

갱신 비율의 정의는 다음과 같다.

```
r_k = L_k / S_k   (S_k < L_k)
    = U_k / S_k   (S_k > U_k)
    = 1.0         (L_k <= S_k <= U_k)
```

가중합이 0 이하인 제약 열은 비율을 정의할 수 없으므로 1.0으로 둔다. 상한이 무한대인 열은 상방 절단을 적용하지 않는다.

### `BoundChecker`

구간 목표를 가지는 제약이다. 생성자는 `{제약 열 이름: 실수 또는 (L, U)}` 형태의 사전을 받으며, 사전의 열거 순서가 곧 제약 열의 순서이다. `from_arrays`는 열 이름과 하한·상한 배열로부터 직접 생성한다. `is_within(weighted_sums)`는 제약 열별로 가중합이 목표 구간 안에 있는지 판정한다.

---

## `tensor.py` — 다차원 교차표

### `TensorFlattener`

| 메서드 | 설명 |
| --- | --- |
| `flatten(tensor)` | C 순서(마지막 차원이 가장 빠르게 변함)로 평탄화한다 |
| `cell_names(dims)` | 차원별 범주 목록으로부터 평탄화 순서에 대응하는 셀 이름을 생성한다 |
| `multi_indices(shape)` | 평탄화된 각 열의 다중 색인을 `(n_cells, n_dims)` 배열로 반환한다 |

평탄화 순서는 `hierarchy.mapper.AggregationMapper.cell_names`와 동일하므로, 목표 텐서의 셀과 속성 행렬의 열이 이름 수준에서 대응한다.

### `TensorTarget`

$N$차원 결합 교차표 목표이다. `dims`는 차원 이름에서 범주 목록으로의 대응이며 선언 순서가 텐서의 축 순서이다. 목표 텐서의 형태가 차원 정의와 어긋나면 생성 시점에 예외가 발생한다. `upper` 텐서를 함께 주면 구간 목표가 된다.

| 요소 | 설명 |
| --- | --- |
| `dims`, `shape`, `target_tensor` | 차원 정의와 원본 텐서 |
| `dim_names` | 축 순서대로의 차원 이름 |
| `multi_index_of(column)` | 평탄화된 제약 열이 대응하는 차원별 범주 조합을 반환한다 |
| `category_frame()` | 제약 열별 범주 조합을 표로 반환한다. `LogicalRuleParser`의 입력으로 쓴다 |

`interval_tensor_target(name, level, dims, intervals)`는 셀별 목표가 스칼라와 구간으로 섞여 있는 경우의 생성 보조 함수이다.

---

## `rules.py` — 도메인 규칙 해석

### `LogicalRuleParser`

불가능한 조합을 기술하는 규칙 문자열 목록을 받는다. 평가의 대상은 표본 자료의 행이 아니라 제약 열이 나타내는 범주 조합이다.

| 메서드 | 설명 |
| --- | --- |
| `build_validity_mask(category_frame)` | 길이 $K$의 유효 마스크를 반환한다. 유효한 쪽이 참이다 |
| `structural_zero_columns(category_frame)` | 구조적 영으로 판정된 제약 열의 색인을 반환한다 |
| `describe(category_frame)` | 규칙별로 몇 개의 제약 열이 걸러지는지 정리한 표를 반환한다 |

`category_frame`의 각 행이 제약 열 하나에 대응하며, 행 순서는 속성 행렬의 열 순서와 같아야 한다. `TensorTarget.category_frame()`이 이 형식을 산출한다. 평가 결과가 논리값이 아니거나 길이가 어긋나면 예외가 발생한다.

`DataFrame.eval`은 임의 표현식을 평가하므로, 신뢰할 수 없는 출처의 규칙 문자열을 그대로 넘기지 않도록 호출 지점에서 통제한다.

---

## `zeros.py` — 영 셀 처리

두 클래스의 마스크는 모두 제약 열 축(길이 $K$)을 따르며, 유효한 쪽이 참이다.

### `StructuralZeroMask`

| 메서드 | 설명 |
| --- | --- |
| `check_axis(validity_mask, n_columns)` | 마스크의 길이가 제약 열 수와 같은지 확인한다. 기본 단위 축의 마스크를 넘기면 예외가 발생한다 |
| `select_valid_columns(matrix, validity_mask)` | 유효 제약 열만 남긴 행렬과 남은 열의 원래 색인을 반환한다 |
| `enforce_zero_targets(lower, upper, validity_mask, policy="force_zero")` | 구조적 영 위치에 양수 목표가 들어온 경우를 0으로 강제하거나(`force_zero`) 예외로 처리한다(`error`) |

구조적 영의 색인 원천 제외는 `matrix.UnifiedSparseMatrixBuilder`가 블록 결합 이전에 수행한다. 본 클래스는 이미 구축된 행렬에 같은 처리를 적용할 때 사용한다.

### `ZeroCellResolver`

| 메서드 | 설명 |
| --- | --- |
| `detect(matrix, weights, lower, validity_mask=None)` | 가중합이 0이면서 하한이 양수인 유효 제약 열의 색인을 반환한다 |
| `apply_epsilon_smoothing(matrix, columns, epsilon=1e-5, rows=None)` | 지정한 제약 열에만 $\varepsilon$를 주입한다 |
| `resolve(matrix, weights, lower, validity_mask=None, epsilon=1e-5)` | 탐지와 평활을 연달아 수행하고 대상 열 목록을 함께 반환한다 |

평활은 조밀 변환 없이 좌표 형식의 증분 행렬을 더하는 방식이다. 추가되는 비영 성분의 수는 대상 열의 수와 주입 대상 행의 수의 곱이며, `rows`를 지정하면 일부 기본 단위에만 주입하여 이를 더 줄일 수 있다. 구조적 영은 유효 마스크에 의해 탐지 단계에서 제외되므로 $\varepsilon$가 주입되지 않는다.

---

## `missing.py` — 결측 주변표의 구간 추정

공표된 주변표에는 값이 없는 셀이 흔히 존재한다. 비공개 처리, 반올림 공표, 표본 오차로 인한 미공표가 그 원인이다. 결측 셀을 버리면 그 범주가 제약 없이 방치되고, 임의의 점 추정으로 채우면 근거 없는 정보를 주입하게 된다. 본 모듈은 결측 셀에 목표 구간 $[L_k, U_k]$를 부여하고, 그 구간을 관측된 셀과 총계가 함께 만족해야 하는 균형식으로부터 연역한다.

### 균형식 (`LinearBalance`)

**정의.** 균형식은 셀 값들이 만족해야 하는 선형 조건이다.

$$\sum_{k} c_k x_k \in [T_{lo}, T_{hi}]$$

| 필드 | 설명 |
| --- | --- |
| `name` | 균형식의 이름 |
| `coefficients` | 셀 이름에서 계수 $c_k$로의 대응. 모든 계수는 양수여야 한다 |
| `total` | 균형식의 값이 놓일 구간. 총계가 정확히 알려지면 퇴화 구간이다 |

계수를 모두 1로 두면 주변표의 총계 조건이 된다. 계수를 달리 두면 표 사이의 정합 조건도 같은 틀로 표현된다.

| 조건 | 균형식 |
| --- | --- |
| 주변표 총계 | $\sum_k x_k = T$ |
| $n$인가구 수와 총 가구원 수 | $\sum_n n \cdot H_n = P$ |
| 가구 수 계급별 거처 수와 총 가구 수 | $\sum_m m \cdot D_m = H$ |

계수가 모두 양수여야 하는 이유는 구간 전파의 단조성이 부호에 의존하기 때문이다.

### `MissingMarginEstimator`

`cells`는 셀 이름에서 관측값으로의 대응이며, 값의 형태가 셀의 상태를 규정한다.

| 입력 형태 | 뜻 | 초기 구간 |
| --- | --- | --- |
| 스칼라 | 정확히 공표된 값 | 퇴화 구간 $[v, v]$ |
| `(하한, 상한)` | 반올림 공표나 비공개 상한 | 주어진 구간 |
| `None` | 값이 없는 결측 셀 | `default_bounds` (기정값 $[0, \infty)$) |

| 메서드 | 설명 |
| --- | --- |
| `add_balance(name, total, coefficients=None)` | 균형식을 등록한다. `coefficients`를 생략하면 모든 셀의 계수를 1로 둔다 |
| `estimate(max_passes=100, tolerance=1e-9)` | 셀 이름에서 `IntervalTarget`으로의 대응을 반환한다 |
| `allocate(balance_name, shares=None)` | 잔차를 결측 셀에 배분한 점 추정을 반환한다. 생략하면 균등 배분한다 |
| `to_constraint(name, level)` | 셀별 추정 구간을 `BoundChecker`로 만든다. 열 순서는 `cells`의 선언 순서이다 |
| `balance_constraint(balance_name, name, level)` | 균형식의 총계 자체를 제약 열 하나로 만든다 |
| `balance_hierarchy(balance_name)` | 균형식의 셀들을 하나의 대범주로 묶은 `CategoryHierarchy`를 반환한다 |
| `to_frame()` | 셀별 관측 상태와 추정 구간, 남은 불확실성 폭을 표로 정리한다 |
| `missing_cells`, `observed_cells` | 결측 셀과 관측 셀의 이름 목록 |

### 구간 전파

균형식 하나와 셀별 현재 구간이 주어지면, 셀 $k$의 구간은 다음으로 좁혀진다.

$$U_k \leftarrow \min\left(U_k,\ \frac{T_{hi} - \sum_{j \neq k} c_j L_j}{c_k}\right), \qquad L_k \leftarrow \max\left(L_k,\ \frac{T_{lo} - \sum_{j \neq k} c_j U_j}{c_k}\right)$$

균형식이 하나뿐이면 한 번의 순회로 최적 구간에 도달한다. 여러 균형식이 셀을 공유하면 한 균형식에서 좁혀진 구간이 다른 균형식의 입력이 되므로 순회를 반복한다. 이때 구간은 유한 회차에 닫히지 않고 점근적으로 좁혀질 수 있으며, 어디까지 좁힐지는 `tolerance`와 `max_passes`가 결정한다. 순회마다 구간이 단조 감소하므로 절차 자체는 항상 정지한다.

좁힌 결과 $L_k > U_k$가 되면 관측값과 총계가 서로 모순된다는 뜻이므로 예외가 발생한다. 이는 목표 정합성의 사전 검사 역할도 겸한다.

### 결합 조건의 부과

셀별 구간만으로는 "결측 셀들의 합이 잔차와 같다"는 결합 조건이 표현되지 않는다. 예를 들어 잔차가 70이고 결측 셀이 둘이면 각 셀의 구간은 $[0, 70]$이지만, 두 셀이 동시에 70이 되는 것을 셀별 구간은 막지 못한다. 결합 조건까지 부과하려면 다음 두 가지를 함께 등록한다.

1. `balance_hierarchy`가 반환한 위계로 `CoarseBlock`을 추가하여 셀들의 합을 세는 제약 열을 만든다.
2. `balance_constraint`가 만든 총계 제약을 등록소에 등록한다.

매핑 행렬은 이진 행렬이므로 `balance_hierarchy`는 계수가 모두 1인 균형식에만 적용할 수 있다. 계수가 1이 아닌 균형식은 계수를 반영한 속성 열을 `ColumnBlock`으로 직접 구축한다.

### 보조 함수

| 함수 | 설명 |
| --- | --- |
| `marginal_from_series(values, missing_as=None)` | 결측(`NaN`)을 담은 주변표 `Series`를 추정기의 입력 형태로 변환한다 |
| `balance_from_classes(class_labels, class_sizes)` | 계급별 개체 수와 총 개체 수를 잇는 균형식의 계수를 만든다 |

상한이 없는 계급(`'4+'` 등)은 대표값을 직접 지정해야 하며, 그 대표값의 오차는 균형식의 총계를 구간으로 주어 흡수한다.

---

## `registry.py` — 제약 등록소

### `ConstraintRegistry`

제약 객체를 등록 순서대로 보관하고, 각 제약이 점유하는 제약 열 구간을 관리한다. 속성 행렬의 열 배치에 대한 단일 기준점이다.

| 메서드 및 속성 | 설명 |
| --- | --- |
| `register(constraint)`, `extend(constraints)` | 제약을 등록한다. 이름이 중복되면 예외가 발생한다 |
| `get_all()`, `get(name)` | 등록된 제약을 조회한다 |
| `slice_of(name)` | 해당 제약이 점유하는 열 구간을 반환한다 |
| `owner_of_column(column)` | 제약 열이 속한 제약의 이름을 반환한다 |
| `n_columns`, `column_names`, `column_owners` | 전체 제약 열의 수, `제약명::열이름` 형식의 이름, 열별 소유 제약명 |
| `bounds()` | 등록 순서대로 이어 붙인 하한과 상한 배열을 반환한다 |
| `compute_ratios(weighted_sums)` | 제약별로 비율을 산출하여 이어 붙인다 |
| `validate_against(n_matrix_columns)` | 속성 행렬의 열 수와 등록된 제약 열 수가 일치하는지 확인한다 |

등록소를 엔진에 넘기면 비율 산출이 제약별로 위임되므로, 제약마다 다른 갱신 규칙을 정의할 수 있다. 넘기지 않으면 엔진이 전체 제약 열에 구간별 기정 규칙을 일괄 적용한다.
