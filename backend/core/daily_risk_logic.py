from __future__ import annotations
from typing import List, Dict, Any, Optional
import pandas as pd
from core.config import (
    MERGED_MONTHLY_PATH,
    MONTHLY_PREDICT_VS_ACTUAL_PATH,
)
from modeling.daily_risk_index.monthly_tree_and_daily_risk import (
    load_monthly_predictions,
    get_daily_risk,
)
from core.external_env_api import build_airkorea_env_obs_for_all_sido

# ─────────────────────────────────────
# 1) 전역 캐시: 월 예측값, 기준 환경값(ref_df)
# ─────────────────────────────────────

_monthly_preds_df: Optional[pd.DataFrame] = None
_ref_df: Optional[pd.DataFrame] = None


def _load_monthly_preds(reload: bool = False) -> pd.DataFrame:
    """
    monthly_predict_vs_actual.csv 를 로드해서 캐싱한다.
    """
    global _monthly_preds_df

    if (_monthly_preds_df is not None) and (not reload):
        return _monthly_preds_df

    # monthly_tree_and_daily_risk.load_monthly_predictions 를 그대로 사용
    df = load_monthly_predictions(MONTHLY_PREDICT_VS_ACTUAL_PATH)

    # 기본 형태 체크
    required_cols = {"year", "month", "sido_name", "y_hat"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(
            f"monthly_predict_vs_actual.csv 에 필요한 컬럼이 없습니다: {missing}"
        )

    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)

    _monthly_preds_df = df
    return _monthly_preds_df


def _build_ref_df_from_merged() -> pd.DataFrame:
    """
    data/mart/merged_monthly.csv 를 이용해서
    (sido_name, month) 기준 월별 평균 환경값 ref_df 를 만든다.

    get_daily_risk() 가 기대하는 컬럼 이름에 맞춰서 생성:
      - pm25_month_mean
      - pm10_month_mean
      - t_month_mean
      - rh_month_mean
    """
    df = pd.read_csv(MERGED_MONTHLY_PATH)

    required_cols = {
        "sido_name",
        "month",
        "pm25_mean",
        "pm10_mean",
        "t_mean_month",
        "rh_mean_month",
    }
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"merged_monthly.csv 에 필요한 컬럼이 없습니다: {missing}")

    # month는 int로 통일
    df["month"] = df["month"].astype(int)

    # 시도×월 기준으로 평균값 계산
    ref = df.groupby(["sido_name", "month"], as_index=False).agg(
        pm25_month_mean=("pm25_mean", "mean"),
        pm10_month_mean=("pm10_mean", "mean"),
        t_month_mean=("t_mean_month", "mean"),
        rh_month_mean=("rh_mean_month", "mean"),
    )

    return ref


def _load_ref_df(reload: bool = False) -> pd.DataFrame:
    """
    ref_df (시도×월별 기준 환경값 테이블)를 로드/생성해서 캐싱한다.

    지금은 merged_monthly.csv 기반으로 직접 만든다.
    나중에 별도 ref CSV를 쓰고 싶으면 이 부분만 교체하면 됨.
    """
    global _ref_df

    if (_ref_df is not None) and (not reload):
        return _ref_df

    _ref_df = _build_ref_df_from_merged()
    return _ref_df


# ─────────────────────────────────────
# 2) 날짜별 일일 위험지수 DataFrame 계산
# ─────────────────────────────────────


def compute_daily_risk_df(
    date_str: str,
    env_obs_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    특정 날짜(date_str)에 대해 시도별 일 단위 위험지수를 계산한다.

    Parameters
    ----------
    date_str : str
        'YYYY-MM-DD' 형식의 문자열 (예: "2024-01-01")
    env_obs_df : pd.DataFrame, optional
        시도별 당일 환경 관측값.
        - 기본 스키마 예시: ['sido_name', 'pm25', 'pm10', 't', 'rh']
        - None 인 경우, 함수 내부에서 에어코리아 실시간 API를 호출하여
          전 시도에 대한 pm25/pm10 평균 값을 조회한 뒤 사용한다.
        - AirKorea 레이트 리밋(429) 등으로 인해 env_obs_df 를 얻지 못한 경우,
          env_obs_df=None 으로 get_daily_risk 를 호출하여 기준 ref_df 만 사용하는
          모드로 계산을 진행한다.

    Returns
    -------
    pd.DataFrame
        monthly_tree_and_daily_risk.get_daily_risk() 가 반환하는 DataFrame.
        기본 컬럼:
          ['date', 'year', 'month', 'ym', 'sido_name',
           'y_hat', 'base_daily', 'w_today', 'y_day_hat',
           'risk_score', 'risk_level']
    """
    monthly_preds_df = _load_monthly_preds()
    ref_df = _load_ref_df()

    # env_obs_df 가 주어지지 않으면 에어코리아 실시간 값으로 자동 조회
    if env_obs_df is None:
        env_obs_df = build_airkorea_env_obs_for_all_sido()
        # AirKorea 레이트 리밋(429) 등으로 인해 None 이 반환될 수 있다.
        if env_obs_df is None:
            # 실시간 환경 데이터 없이, ref_df + 월 예측값만으로 위험지수 계산
            print(
                "[INFO] env_obs_df 없음 → 실시간 대기질 없이 기준값(ref_df)만 사용하여 "
                "일일 위험지수 계산을 진행합니다."
            )

    df = get_daily_risk(
        date_str=date_str,
        monthly_preds_df=monthly_preds_df,
        env_obs_df=env_obs_df,  # None 일 수도 있음 (모델이 이를 허용하도록 설계되어 있음)
        ref_df=ref_df,
        # percentile_clip 기본값 (10, 90) 그대로 사용
    )

    return df


# ─────────────────────────────────────
# 3) /api/daily_risk 용: JSON 변환 헬퍼
# ─────────────────────────────────────


def build_daily_risk_rows_for_api(
    date_str: str,
    env_obs_df: Optional[pd.DataFrame] = None,
) -> List[Dict[str, Any]]:
    """
    /api/daily_risk 엔드포인트에서 사용하기 위한 헬퍼.

    - compute_daily_risk_df(...) 로 DataFrame을 만든 뒤,
    - 프론트에서 바로 사용 가능한 리스트[dict] 형태로 변환한다.

    반환 형식 예:
      [
        {
          "sido_name": "서울",
          "value": 78.5,
          "extra": {
            "risk_score": 78.5,
            "risk_level": "High",
            "date": "2024-01-01"
          }
        },
        ...
      ]
    """
    df = compute_daily_risk_df(date_str=date_str, env_obs_df=env_obs_df)

    # 혹시라도 필요한 경우 대비해서 타입 통일
    df["sido_name"] = df["sido_name"].astype(str)
    df["risk_score"] = df["risk_score"].astype(float)
    df["risk_level"] = df["risk_level"].astype(str)

    rows: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        rows.append(
            {
                "sido_name": row["sido_name"],
                "value": float(row["risk_score"]),  # 맵 색칠용 값
                "extra": {
                    "risk_score": float(row["risk_score"]),
                    "risk_level": row["risk_level"],
                    "date": row["date"],
                    # 필요하면 y_day_hat / w_today 등도 여기 넣을 수 있음
                    # "y_day_hat": float(row["y_day_hat"]),
                    # "w_today": float(row["w_today"]),
                },
            }
        )

    return rows
