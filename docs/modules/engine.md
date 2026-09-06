# `generalized_ipu.engine` — 반복 수렴 연산 엔진

가중치의 반복 갱신을 주도하고, 극단값을 억제하며, 종료 시점을 판정하는 패키지이다. `matrix`, `constraints`, `diagnostics`에 의존한다.

**책임 범위**

- 반복 루프를 주도하고 종료 조건을 판정한다.
- 갱신 결과의 극단값을 억제한다.
- 수렴 지표를 산출하고 회차별 이력을 기록한다.

**책임 밖**

- 갱신 비율의 산출 규칙은 `constraints`가 정의한다.
- 희소 연산 자체는 `matrix`가 수행한다.

---

## `core.py` — 반복 루프

### `GeneralizedIPUEngine`

목표는 하한과 상한의 쌍으로 받는다. `upper`를 생략하면 스칼라 목표로 보아 `lower`와 동일하게 둔다.

| 인자 | 기정값 | 의미 |
| --- | --- | --- |
| `matrix` | — | 속성 행렬 $A$ (`csr_matrix`, $n \times K$) |
| `lower`, `upper` | — | 제약 열별 목표 구간. 길이는 $K$ |
| `max_iterations` | 100 | 최대 반복 회차 |
| `relative_gap` | 0.01 | 수렴 허용 상대 격차 |
| `absolute_diff` | 0.0 | 수렴 허용 절대 격차 |
| `weight_floor` | $10^{-5}$ | 가중치의 절대 하한 |
| `min_ratio`, `max_ratio` | 0.1, 10.0 | 회차별 변화 배율의 하한과 상한 |
| `eta` | 1.0 | 완화 계수. $(0, 1]$ 범위 |
| `registry` | `None` | 제약 등록소. 주면 비율 산출을 제약별로 위임한다 |
| `tracker` | `None` | 반복 이력 수집기. 생략하면 새로 생성한다 |

생성 시점에 목표의 길이와 속성 행렬의 열 수가 일치하는지, 하한이 상한을 넘지 않는지, 완화 계수가 허용 범위인지 확인한다.

**생성 보조**

| 메서드 | 설명 |
| --- | --- |
| `from_registry(matrix, registry, **kwargs)` | 등록소의 목표를 그대로 사용한다. 열 수 정합성도 함께 확인한다 |
| `from_targets(matrix, targets, **kwargs)` | 스칼라와 구간이 섞인 목표 나열을 정규화하여 생성한다 |

### `fit(initial_weights=None) -> IPUResult`

초기 가중치를 지정하지 않으면 모든 성분이 1인 벡터에서 출발한다. 지정한 경우 복사본을 사용하므로 호출자의 배열은 변경되지 않으며, 양수가 아닌 성분이 있으면 예외가 발생한다.

**한 회차의 처리 순서**

1. 갱신 비율을 산출한다. 등록소가 있으면 제약별로 위임하고, 없으면 구간별 기정 규칙을 일괄 적용한다.
2. 동시 갱신식으로 새 가중치 후보를 계산한다.
3. 배율 제한과 가중치 하한을 적용한다.
4. 가중합을 다시 계산하고 수렴을 판정한다.
5. 이력을 기록한다.

반복에 들어가기 전에 초기 가중치에 대한 판정과 기록을 먼저 수행하므로, 초기 가중치가 이미 수렴 조건을 만족하면 회차 0으로 종료한다. 이력의 행 수는 항상 `n_iterations + 1`이다.

### `IPUResult`

| 속성 | 설명 |
| --- | --- |
| `weights` | 최종 가중치 |
| `converged` | 수렴 여부 |
| `termination_reason` | `"converged"` 또는 `"iteration_limit_reached"` |
| `n_iterations` | 실제 수행한 회차 |
| `report` | 최종 `ConvergenceReport` |
| `weighted_sums` | 최종 가중합 |
| `tracker` | 회차별 이력 |

수렴과 최대 반복 도달은 다른 종료 사유이므로 `termination_reason`으로 구분하여 보고한다.

`engine.inactive_units`는 어떤 제약에도 관여하지 않아 가중치가 초기값에 머무르는 기본 단위의 색인을 반환한다.

---

## `bounds.py` — 가중치 경계 제어

### `WeightBoundController`

| 인자 | 기정값 | 의미 |
| --- | --- | --- |
| `weight_floor` | $10^{-5}$ | 갱신 후 가중치의 절대 하한 |
| `min_ratio` | 0.1 | 한 회차 변화 배율의 하한 |
| `max_ratio` | 10.0 | 한 회차 변화 배율의 상한 |

`apply_bounds(old_weights, new_weights)`의 처리 순서는 다음과 같다.

1. 변화 배율을 계산한다. 분모는 `DENOMINATOR_FLOOR`($10^{-10}$)로 보호한다.
2. 배율을 `[min_ratio, max_ratio]`로 절단한다.
3. 이전 가중치에 절단된 배율을 곱한다.
4. 결과에 가중치 하한을 적용한다.

절단 대상은 가중치의 절대 수준이 아니라 회차별 변화 배율이다. 한 회차의 이동 폭을 제한함으로써 특정 기본 단위가 소수의 제약을 홀로 충족하려고 급격히 팽창하거나 소멸하는 현상을 막는다. 여러 회차에 걸쳐 같은 방향의 조정이 누적되면 가중치의 절대 수준은 여전히 크게 변할 수 있으며, 이는 의도된 동작이다. 절대 수준을 제한하려면 별도의 수준 제한 장치가 필요하다.

배율 절단은 갱신 방향을 유지한 채 보폭만 줄이므로 수렴 방향을 왜곡하지 않는다. 반면 가중치 하한은 0에 접근하는 가중치를 강제로 들어올리므로 가중합에 계통적 상향 편의를 유발한다. `last_statistics`의 `n_ratio_clipped`와 `n_floor_clipped`가 회차마다 개입 건수를 기록하며, 이 값은 이력에 함께 남는다.

---

## `convergence.py` — 수렴 판정

### `ConvergenceEvaluator`

`evaluate(weighted_sums, lower, upper=None) -> ConvergenceReport`가 구간 위반량 $v_k = \max(L_k - S_k,\ S_k - U_k,\ 0)$을 기준으로 판정한다.

제약 열은 절대 격차와 상대 격차 중 하나만 충족해도 만족한 것으로 본다.

- 절대 격차 조건: $v_k \le$ `absolute_diff`
- 상대 격차 조건: $v_k / \max(L_k, 1) \le$ `relative_gap`

`absolute_diff`를 0으로 두면 상대 격차만으로 판정한다. 목표값이 작은 제약 열에서 상대 격차가 과대 평가되는 경우 `absolute_diff`를 함께 지정한다.

### `ConvergenceReport`

| 필드 | 의미 |
| --- | --- |
| `converged` | 모든 제약 열이 만족되었는가 |
| `max_relative_gap` | 상대 격차의 최댓값 |
| `max_absolute_diff` | 구간 위반량의 최댓값 |
| `violation_rate` | 미충족 제약 열의 비율 |
| `n_violations` | 미충족 제약 열의 수 |

최댓값 하나만이 아니라 위반율을 함께 산출하므로, 소수의 제약 열이 판정을 좌우하는지 다수가 미충족인지 구분할 수 있다.

목표가 0인 제약 열도 판정 대상이다. $L_k = U_k = 0$인데 가중합이 양수이면 위반량이 그대로 양수가 되므로, 구조적 영 처리가 실패한 경우가 수렴으로 보고되지 않는다.
