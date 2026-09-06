# `generalized_ipu.diagnostics` — 진단과 보고

반복 이력을 수집하고 적합도와 가중치 분포를 보고하는 패키지이다. 배열과 표만 입력으로 받으며, 다른 내부 패키지에 의존하지 않는다. 이 독립성은 의도된 설계이며, 진단 모듈이 연산 경로에 영향을 주지 않도록 유지한다.

**책임 범위**

- 회차별 수렴 지표의 이력을 축적한다.
- 제약 열별 적합도를 표로 산출한다.
- 최종 가중치 분포의 건전성을 검증한다.

**책임 밖**

- 반복 제어에 관여하지 않는다. 진단 결과가 종료 조건을 바꾸지 않는다.

---

## `tracker.py` — 반복 이력 수집

### `IterationRecord`

한 회차의 기록이다. 회차 번호, 최대 상대 격차, 최대 구간 위반량, 구간 위반율, 미충족 제약 열 수, 배율 절단 건수, 하한 접촉 건수를 담는다.

### `ConvergenceTracker`

| 요소 | 설명 |
| --- | --- |
| `log(iteration, report, n_ratio_clipped=0, n_floor_clipped=0)` | 한 회차를 기록하고 그 기록을 반환한다 |
| `records` | 기록 목록 |
| `history` | 회차별 최대 상대 격차만 뽑은 목록 |
| `to_frame()` | 기록 전체를 `DataFrame`으로 변환한다. 기록이 없으면 빈 표를 반환한다 |
| `diagnose(tolerance=1e-12)` | 이력의 형태로부터 진단 소견을 문자열로 반환한다 |

엔진이 회차마다 `log`를 호출한다. 초기 가중치에 대한 판정이 회차 0으로 기록되므로 표의 행 수는 `n_iterations + 1`이다.

**이력 형태의 해석**

| 형태 | 소견 |
| --- | --- |
| 단조 감소 후 평탄화 | 정상 수렴 |
| 일정 수준에서 진동 | 과잉 제약 또는 완화 계수 과대 |
| 감소가 매우 느림 | 완화 계수 과소, 또는 제약 간 상충 |
| 증가 | 갱신식 또는 목표 정규화의 결함 |

---

## `reporter.py` — 적합도 보고서

### `FitReportGenerator`

#### `generate_report(weighted_sums, lower, upper=None, column_names=None, column_owners=None, relative_gap=0.0, absolute_diff=0.0) -> pd.DataFrame`

제약 열별 적합도를 상대 격차의 내림차순으로 정렬하여 반환한다.

| 열 이름 | 의미 |
| --- | --- |
| `constraint` | 소속 제약명. `column_owners`를 준 경우에만 생성한다 |
| `column` | 제약 열 이름 |
| `lower`, `upper` | 목표 구간 |
| `estimated` | 최종 가중합 |
| `bound_violation` | 구간 위반량 $v_k$ |
| `relative_gap` | $v_k / \max(L_k, 1)$ |
| `within_bounds` | 위반량이 0인가. 허용 한계를 반영하지 않는 엄격한 판정 |
| `satisfied` | `relative_gap`, `absolute_diff` 허용 한계를 반영한 판정 |

`relative_gap`과 `absolute_diff` 인자는 `ConvergenceEvaluator`와 같은 규칙으로 `satisfied` 열을 산출한다. 엔진에 지정한 값과 같게 두면 보고서의 판정이 엔진의 수렴 판정과 일치한다. 기정값 0에서는 `satisfied`가 `within_bounds`와 같아진다.

정렬이 내림차순이므로 표의 상단이 곧 적합도가 가장 나쁜 제약 열이다. 미수렴을 진단할 때는 이 상단 항목들이 특정 계층이나 특정 속성에 몰려 있는지를 먼저 확인한다. `column_names`는 `AttributeMatrix.column_names` 또는 `ConstraintRegistry.column_names`에서 얻는다.

#### `summarize_by_constraint(report) -> pd.DataFrame`

제약별로 열 수, 구간을 벗어난 열 수(`n_outside_bounds`), 허용 한계까지 반영한 미충족 열 수(`n_unsatisfied`), 최대 상대 격차, 최대 구간 위반량, 목표 총계, 추정 총계를 집계한다. `column_owners`를 지정하여 생성한 보고서에만 적용된다.

#### `worst_columns(report, n=10) -> pd.DataFrame`

적합도가 가장 나쁜 제약 열 상위 $n$개를 반환한다.

---

## `distribution.py` — 가중치 분포 분석

제약 적합도가 양호하더라도 소수의 기본 단위가 전체 가중치의 대부분을 차지하면, 합성된 모집단은 표본의 다양성을 잃고 소수 개체의 복제물이 된다. 적합도만으로는 이 상태를 탐지할 수 없다.

### `WeightDistributionAnalyzer.analyze(weights, initial_weights=None, weight_floor=None)`

`WeightDistributionReport`를 반환한다.

| 지표 | 의미 |
| --- | --- |
| `minimum`, `maximum`, `mean`, `std`, `quantiles` | 기술 통계. 분위수는 1, 25, 50, 75, 99 백분위 |
| `coefficient_of_variation` | 표준편차를 평균으로 나눈 값 |
| `normalized_entropy` | 가중치 분포의 균등성. 1에 가까울수록 균등 |
| `effective_sample_size` | $(\sum_i w_i)^2 / \sum_i w_i^2$ |
| `top_1_percent_share`, `top_5_percent_share` | 상위 기본 단위가 차지하는 가중치 비중 |
| `n_at_floor` | 가중치 하한에 걸린 기본 단위의 수. `weight_floor`를 준 경우에만 산출한다 |
| `max_expansion_ratio`, `min_expansion_ratio` | 초기 가중치 대비 최종 가중치 배율. `initial_weights`를 준 경우에만 산출한다 |

`to_series()`는 지표를 평탄한 `Series`로 변환한다.

### `WeightDistributionAnalyzer.flag_concerns(report, min_effective_ratio=0.5, max_top_share=0.5)`

주의가 필요한 항목을 문자열 목록으로 반환한다. 유효 표본 수가 기본 단위 수에 비해 현저히 작으면 제약을 완화하거나 배율 제한을 강화하거나 표본을 보강해야 한다는 신호로 해석한다.
