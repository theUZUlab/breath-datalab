from __future__ import annotations
from typing import Literal, List, Dict, Any, Optional
import pandas as pd
from backend.core.config import MONTHLY_PREDICT_VS_ACTUAL_PATH


# ─────────────────────────────────────
# 1) 전역 캐시: 월 단위 예측 vs 실제 DataFrame
# ─────────────────────────────────────

_monthly_df: Optional[pd.DataFrame] = None


def load_monthly_predict_vs_actual(reload: bool = False) -> pd.DataFrame:
    """
    data/mart/monthly_predict_vs_actual.csv 를 로드하는 함수.

    - 서버 기동 후 최초 1회만 디스크에서 읽고,
      이후에는 메모리에 캐싱된 DataFrame을 재사용한다.
    - reload=True 로 호출하면 강제 재로딩한다.
    """
    global _monthly_df

    if (_monthly_df is not None) and (not reload):
        return _monthly_df

    df = pd.read_csv(MONTHLY_PREDICT_VS_ACTUAL_PATH)

    # 기본 컬럼 체크
    required_cols = {
        "year",
        "month",
        "sido_name",
        "resp_rate_total",
        "y_hat",
        "error_abs",
        "error_pct",
    }
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(
            f"monthly_predict_vs_actual.csv 에 필요한 컬럼이 없습니다: {missing}"
        )

    # year, month는 int로 캐스팅해 두는 것이 안전하다.
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)

    _monthly_df = df
    return _monthly_df


# ─────────────────────────────────────
# 2) 사용 가능한 연/월 정보 조회 (선택 기능)
# ─────────────────────────────────────


def get_available_year_months() -> List[Dict[str, int]]:
    """
    프론트엔드에서 드롭다운 옵션을 만들 때 사용할 수 있는
    (year, month) 조합 리스트를 반환한다.

    예:
        [
            {"year": 2018, "month": 1},
            {"year": 2018, "month": 2},
            ...
        ]
    """
    df = load_monthly_predict_vs_actual()
    grouped = (
        df[["year", "month"]]
        .drop_duplicates()
        .sort_values(["year", "month"])
        .reset_index(drop=True)
    )

    return [
        {"year": int(row["year"]), "month": int(row["month"])}
        for _, row in grouped.iterrows()
    ]


# ─────────────────────────────────────
# 3) /api/monthly_map 에서 사용할 데이터 가공 함수
# ─────────────────────────────────────

MetricType = Literal["actual", "pred", "error_pct"]


def get_monthly_map_rows(
    year: int,
    month: int,
    metric: MetricType = "actual",
) -> List[Dict[str, Any]]:
    """
    특정 연/월에 대한 시도별 월 단위 값을 맵 시각화용 리스트로 반환한다.

    Parameters
    ----------
    year : int
        조회할 연도 (예: 2018, 2019, ...)
    month : int
        조회할 월 (1~12)
    metric : {"actual", "pred", "error_pct"}
        value에 매핑할 지표 종류
        - "actual"    → resp_rate_total
        - "pred"      → y_hat
        - "error_pct" → error_pct

    Returns
    -------
    List[Dict[str, Any]]
        [
          {
            "sido_name": "서울",
            "value": 123.4,
            "extra": {
              "actual": 120.0,
              "pred": 125.0,
              "error_pct": 4.2,
              "year": 2023,
              "month": 1,
              "ym": "2023-01"
            }
          },
          ...
        ]

    Raises
    ------
    ValueError
        - metric 값이 허용 범위를 벗어난 경우
        - 해당 year, month 조합 데이터가 존재하지 않는 경우
    """
    df = load_monthly_predict_vs_actual()

    metric_to_col = {
        "actual": "resp_rate_total",
        "pred": "y_hat",
        "error_pct": "error_pct",
    }

    if metric not in metric_to_col:
        raise ValueError(
            f"지원하지 않는 metric 입니다. "
            f"허용값: {list(metric_to_col.keys())}, 입력값: {metric}"
        )

    value_col = metric_to_col[metric]

    # year, month 필터링
    mask = (df["year"] == int(year)) & (df["month"] == int(month))
    sub = df.loc[mask].copy()

    if sub.empty:
        raise ValueError(f"{year}-{month:02d} 에 대한 월 단위 데이터가 없습니다.")

    # ym 컬럼이 없으면 만들어 둔다.
    if "ym" not in sub.columns:
        sub["ym"] = (
            sub["year"].astype(str) + "-" + sub["month"].astype(str).str.zfill(2)
        )

    rows: List[Dict[str, Any]] = []

    for _, row in sub.iterrows():
        actual = float(row["resp_rate_total"])
        pred = float(row["y_hat"])
        error_pct = float(row["error_pct"])
        sido_name = str(row["sido_name"])
        ym = str(row["ym"])

        # 프론트 색상 맵핑에 쓸 값
        value = {
            "resp_rate_total": actual,
            "y_hat": pred,
            "error_pct": error_pct,
        }[value_col]

        rows.append(
            {
                "sido_name": sido_name,
                "value": float(value),
                "extra": {
                    "actual": actual,
                    "pred": pred,
                    "error_pct": error_pct,
                    "year": int(row["year"]),
                    "month": int(row["month"]),
                    "ym": ym,
                },
            }
        )

    return rows
