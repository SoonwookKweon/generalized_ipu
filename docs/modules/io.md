# `generalized_ipu.io` — 자료 적재와 결과 내보내기

외부 자료를 라이브러리 안으로 들이고, 산출된 가중치를 밖으로 내보내는 경계 패키지이다. 최상위 조립 지점이므로 다른 모든 패키지를 사용할 수 있다.

**책임 범위**

- 계층별 원본 자료를 적재하고 키 열의 자료형을 통일한다.
- 최종 가중치를 원본 자료에 결합하여 저장한다.

**책임 밖**

- 자료의 의미 검증은 `hierarchy`가 담당한다. 본 패키지는 형식 변환과 결합만 수행한다.

---

## `loaders.py` — 자료 적재

### `DataLoader`

| 메서드 | 설명 |
| --- | --- |
| `load_csv(file_path, key_cols=(), attribute_cols=(), key_dtype="string", **kwargs)` | CSV를 적재하고 정규화한다 |
| `load_parquet(file_path, ...)` | Parquet을 적재하고 정규화한다 |
| `normalize(frame, key_cols=(), attribute_cols=(), key_dtype="string")` | 이미 적재한 표를 정규화한다 |
| `build_tree(base_level, levels)` | 계층 설정 사전으로부터 `HierarchyTree`를 구성하고 무결성을 검증한다 |

**정규화 규약**

- 키 열은 명시한 자료형으로 변환한다. 계층마다 키가 정수와 문자열로 다르게 읽히면 결합이 조용히 실패하므로 반드시 통일한다.
- 키 열에 결측이 있으면 예외를 발생시킨다.
- 속성 열은 수치로 변환하고 결측을 0으로 채운다.
- 지정한 열이 자료에 없으면 예외를 발생시킨다.

`build_tree`의 `levels`는 계층명에서 설정으로의 대응이며, 각 설정은 `data`, `primary_key`, `parent_key`, `parent_level`을 가진다.

---

## `exporter.py` — 결과 내보내기

### `DatasetExporter`

| 메서드 | 설명 |
| --- | --- |
| `attach_weights(frame, weights, weight_column="ipu_weight", keys=None, key_column=None)` | 가중치를 표에 붙인다 |
| `propagate_to_level(tree, target_level, weights, weight_column="ipu_weight")` | 기본 계층의 가중치를 하위 계층 자료로 전파한다 |
| `export(frame, output_path, fmt=None, **kwargs)` | CSV 또는 Parquet으로 저장한다. `fmt`를 생략하면 확장자로 판정한다 |
| `export_with_weights(frame, weights, output_path, ...)` | 부착과 저장을 연달아 수행한다 |
| `integerize(weights, random_state=None)` | 확률적 반올림으로 정수화한다 |

**결합 방식.** `keys`와 `key_column`을 주면 주키를 기준으로 결합하고, 대응하지 않는 행이 있으면 예외를 발생시킨다. 생략하면 행 순서가 일치한다고 보고 그대로 붙인다. 순서 전제가 어긋나면 오류 없이 잘못된 가중치가 부여되므로 주키 결합을 권장한다. 주키는 `AttributeMatrix.base_keys`에서 얻는다.

**정수화.** 소수부를 성공 확률로 삼는 베르누이 추출을 사용하므로 총합의 기댓값이 보존된다. 재현이 필요하면 `random_state`를 지정한다.

---

## 조립 순서

전 과정은 다음 순서로 조립한다.

1. `DataLoader`로 계층별 자료를 적재하고 `build_tree`로 `HierarchyTree`를 구성한다.
2. 제약을 생성하여 `ConstraintRegistry`에 등록한다.
3. `LogicalRuleParser`로 각 블록의 유효 마스크를 산출한다.
4. `UnifiedSparseMatrixBuilder.build`로 `AttributeMatrix`를 생성한다.
5. `GeneralizedIPUEngine.from_registry`로 반복을 수행한다.
6. `FitReportGenerator`와 `WeightDistributionAnalyzer`로 진단하고, `DatasetExporter`로 저장한다.

하위 패키지들이 서로를 참조하지 않고 최상위에서만 조립되므로, 각 패키지를 독립적으로 시험할 수 있다.
