"""계층 트리의 정의와 무결성 검증.

계층(level)은 동일한 관측 단위를 가지는 하나의 표이며, 계층 트리는 부모키가
상위 계층의 주키를 참조하는 관계로 계층들을 잇는 유향 비순환 그래프이다.
단일 근(root)을 전제하며 순환과 다중 부모를 허용하지 않는다.
"""

from typing import Dict, List, Optional

import pandas as pd


class HierarchyNode:
    """하나의 계층을 주키와 부모키로 식별하여 보관한다.

    parent_key 는 상위 계층 주키를 담은 자기 자신의 열 이름이고, parent_level 은
    그 상위 계층의 이름이다. parent_level 을 생략하면 HierarchyTree 가 주키가
    일치하는 계층을 찾아 추론한다.
    """

    def __init__(
        self,
        name: str,
        data: pd.DataFrame,
        primary_key: str,
        parent_key: Optional[str] = None,
        parent_level: Optional[str] = None,
        copy: bool = True,
    ):
        self.name = name
        self.data = data.copy() if copy else data
        self.primary_key = primary_key
        self.parent_key = parent_key
        self.parent_level = parent_level

    def __repr__(self) -> str:
        return (
            f"HierarchyNode(name={self.name!r}, n_rows={len(self.data)}, "
            f"primary_key={self.primary_key!r}, parent_level={self.parent_level!r})"
        )


class HierarchyTree:
    """계층들의 부모-자식 관계를 관리하고 무결성을 검증한다.

    base_level 은 가중치가 직접 부여되는 기본 계층의 이름이며, 라이브러리가
    추론하지 않고 사용자가 지정한다. 다른 계층의 속성은 모두 이 계층의 행
    좌표계로 옮겨진 뒤 속성 행렬에 편입된다.
    """

    def __init__(self, base_level: str):
        self.base_level = base_level
        self.nodes: Dict[str, HierarchyNode] = {}

    # ------------------------------------------------------------------ 등록

    def add_level(
        self,
        level_name: str,
        data: pd.DataFrame,
        primary_key: str,
        parent_key: Optional[str] = None,
        parent_level: Optional[str] = None,
        copy: bool = True,
    ) -> "HierarchyTree":
        """계층 하나를 등록하고 자기 자신을 반환하여 연쇄 호출을 허용한다.

        이 단계에서는 열의 존재 여부만 확인한다. 참조 무결성과 주키 유일성은
        `validate_integrity` 가 검사한다.
        """
        if level_name in self.nodes:
            raise ValueError(f"계층 '{level_name}' 이 이미 등록되어 있습니다.")
        if primary_key not in data.columns:
            raise KeyError(f"'{level_name}' 계층에 주키 '{primary_key}' 열이 없습니다.")
        if parent_key is not None and parent_key not in data.columns:
            raise KeyError(f"'{level_name}' 계층에 부모키 '{parent_key}' 열이 없습니다.")
        if parent_level is not None and parent_key is None:
            raise ValueError(f"'{level_name}' 계층에 parent_level 만 지정되고 parent_key 가 없습니다.")

        self.nodes[level_name] = HierarchyNode(
            level_name, data, primary_key, parent_key, parent_level, copy=copy
        )
        return self

    # ------------------------------------------------------------------ 조회

    @property
    def base_node(self) -> HierarchyNode:
        """기본 계층의 노드를 반환한다."""
        if self.base_level not in self.nodes:
            raise ValueError(f"기본 계층 '{self.base_level}' 이 등록되어 있지 않습니다.")
        return self.nodes[self.base_level]

    def resolve_parent_level(self, level_name: str) -> Optional[str]:
        """명시된 부모 계층을 반환하고, 없으면 부모키로부터 추론한다.

        추론은 부모키를 주키로 가지는 계층이 정확히 하나일 때만 성립한다.
        후보가 없거나 둘 이상이면 parent_level 을 명시하라는 예외를 발생시킨다.
        최상위 계층이면 None 을 반환한다.
        """
        node = self.nodes[level_name]
        if node.parent_level is not None:
            return node.parent_level
        if node.parent_key is None:
            return None

        candidates = [
            other.name
            for other in self.nodes.values()
            if other.name != level_name and other.primary_key == node.parent_key
        ]
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise ValueError(
                f"'{level_name}' 계층의 부모키 '{node.parent_key}' 를 주키로 가지는 계층이 없습니다. "
                "parent_level 을 명시하십시오."
            )
        raise ValueError(
            f"'{level_name}' 계층의 부모 계층 후보가 여럿입니다: {candidates}. "
            "parent_level 을 명시하십시오."
        )

    def ancestors(self, level_name: str) -> List[str]:
        """해당 계층에서 근(root)까지의 상위 계층 목록을 가까운 순으로 반환한다."""
        path: List[str] = []
        visited = {level_name}
        current = level_name
        while True:
            parent = self.resolve_parent_level(current)
            if parent is None:
                return path
            if parent in visited:
                raise ValueError(f"계층 구조에 순환이 있습니다: {parent}")
            path.append(parent)
            visited.add(parent)
            current = parent

    def path_from_base(self, level_name: str) -> List[str]:
        """기본 계층에서 대상 계층까지 내려가는 경로를 반환한다.

        대상이 기본 계층의 하위가 아니면 빈 목록을 반환한다.
        """
        if level_name == self.base_level:
            return [level_name]
        chain = self.ancestors(level_name)
        if self.base_level not in chain:
            return []
        cut = chain[: chain.index(self.base_level) + 1]
        return list(reversed(cut)) + [level_name]

    def relation_to_base(self, level_name: str) -> str:
        """기본 계층과의 관계를 'base', 'descendant', 'ancestor' 중 하나로 판정한다.

        여기서 상위 계층은 포섭하는 쪽을, 하위 계층은 포섭되는 쪽을 뜻한다.
        기본 계층이 가구이면 거처는 상위 계층이고 가구원은 하위 계층이다.
        """
        if level_name == self.base_level:
            return "base"
        if self.path_from_base(level_name):
            return "descendant"
        if level_name in self.ancestors(self.base_level):
            return "ancestor"
        raise ValueError(f"'{level_name}' 계층이 기본 계층 '{self.base_level}' 과 연결되어 있지 않습니다.")

    # ------------------------------------------------------------------ 검증

    def validate_integrity(self) -> bool:
        """무결성 조건 네 가지를 모두 검사한다.

        기본 계층의 존재, 부모키 열의 존재, 참조 무결성, 주키 유일성을 순서대로
        확인하며, 하나라도 어긋나면 예외를 발생시킨다. 부모키의 결측은 참조
        무결성 위반으로 취급한다.
        """
        if self.base_level not in self.nodes:
            raise ValueError(f"기본 계층 '{self.base_level}' 이 HierarchyTree 에 존재하지 않습니다.")

        for name, node in self.nodes.items():
            # 조건 2: 부모키 열의 존재
            if node.parent_key and node.parent_key not in node.data.columns:
                raise KeyError(f"'{name}' 계층에 부모키 '{node.parent_key}' 열이 존재하지 않습니다.")

            # 조건 4: 주키의 유일성
            duplicated = int(node.data[node.primary_key].duplicated().sum())
            if duplicated:
                raise ValueError(
                    f"'{name}' 계층의 주키 '{node.primary_key}' 에 중복 값이 {duplicated}건 있습니다."
                )

            # 조건 3: 참조 무결성
            if node.parent_key is None:
                continue
            parent_level = self.resolve_parent_level(name)
            if parent_level is None:
                continue
            parent_node = self.nodes[parent_level]
            child_keys = node.data[node.parent_key].dropna()
            unmatched = ~child_keys.isin(set(parent_node.data[parent_node.primary_key]))
            n_unmatched = int(unmatched.sum())
            n_null = int(node.data[node.parent_key].isna().sum())
            if n_unmatched or n_null:
                raise ValueError(
                    f"'{name}' 계층의 부모키 '{node.parent_key}' 중 "
                    f"{n_unmatched}건이 '{parent_level}' 계층의 주키와 대응하지 않고, "
                    f"{n_null}건이 결측입니다."
                )
        return True
