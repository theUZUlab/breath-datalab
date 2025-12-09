import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_absolute_error, mean_squared_error
import calendar
from datetime import datetime
from typing import Optional, Tuple

# ============================================
# 0. 경로 설정 (프로젝트 루트 기준)
# ============================================

# 이 스크립트 위치:
#   backend/modeling/daily_risk_index/monthly_tree_and_daily_risk.py
# 기준으로 세 단계 위가 프로젝트 루트(breath-datalab)라고 가정한다.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# data 폴더 (merged.py와 동일하게 사용)
DATA_DIR = PROJECT_ROOT / "data"

# merged.py 결과 CSV (필요시 여기만 수정)
MART_PATH = DATA_DIR / "mart" / "merged_monthly.csv"

# 모델링 결과, 그래프, 일 단위 위험지수 CSV를 모아둘 폴더
OUTPUT_DIR = PROJECT_ROOT / "data" / "model" / "daily_risk_index"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 월 단위 예측 결과 및 요약 파일 경로
MONTHLY_PRED_PATH = DATA_DIR / "mart" / "monthly_predict_vs_actual.csv"
MONTHLY_DT_RESULTS_PATH = OUTPUT_DIR / "monthly_dt_results.csv"
MONTHLY_ERROR_SUMMARY_PATH = OUTPUT_DIR / "monthly_error_summary.csv"


# ============================================
# 1. 월 단위 회귀트리 학습 및 평가
# ============================================


def train_monthly_tree(
    merged_csv_path: Path = MART_PATH,
    monthly_pred_out: Path = MONTHLY_PRED_PATH,
    dt_results_out: Path = MONTHLY_DT_RESULTS_PATH,
    error_summary_out: Path = MONTHLY_ERROR_SUMMARY_PATH,
) -> pd.DataFrame:
    """
    월 단위 회귀트리 모델 학습 파이프라인.

    - 입력: data/mart/merged_monthly.csv (merged.py 결과)
    - 처리:
        · lag 피처 생성
        · 2018~2022 학습 / 2023 테스트로 하이퍼파라미터 탐색
        · 테스트 RMSE 최소 조합을 사용해 2018~2023 전체 재학습
    - 출력:
        · data/mart/monthly_predict_vs_actual.csv
        · outputs/monthly_dt_results.csv
        · outputs/monthly_error_summary.csv
    """
    print("=== 1) 데이터 로드 ===")
    df = pd.read_csv(merged_csv_path)
    print("입력 데이터 shape:", df.shape)

    # -----------------------------
    # 1-1) 모델 입력 컬럼 선택
    # -----------------------------
    cols_keep = [
        "year",
        "month",
        "ym",
        "sido_name",
        "resp_total",
        "population",
        "resp_rate_total",
        "pm25_mean",
        "pm10_mean",
        "pm25_p90",
        "pm10_p90",
        "pm25_alert_days",
        "pm10_alert_days",
        "t_mean_month",
        "rh_mean_month",
        "precip_sum_month",
        "wind_mean_month",
    ]
    df_model = df[cols_keep].copy()

    # (sido_name, year, month) 기준으로 정렬
    df_model = df_model.sort_values(["sido_name", "year", "month"]).reset_index(
        drop=True
    )

    # -----------------------------
    # 1-2) lag 피처 생성 (이전 달 정보)
    # -----------------------------
    df_model["pm25_mean_lag1m"] = df_model.groupby("sido_name")["pm25_mean"].shift(1)
    df_model["t_mean_month_lag1m"] = df_model.groupby("sido_name")[
        "t_mean_month"
    ].shift(1)
    df_model["resp_rate_total_lag1m"] = df_model.groupby("sido_name")[
        "resp_rate_total"
    ].shift(1)

    # lag NaN 행 제거 (각 시도 첫 달)
    df_model = df_model.dropna(
        subset=["pm25_mean_lag1m", "t_mean_month_lag1m", "resp_rate_total_lag1m"]
    ).reset_index(drop=True)

    # -----------------------------
    # 1-3) 피처/타깃 정의
    # -----------------------------
    feature_cols = [
        "pm25_mean",
        "pm10_mean",
        "pm25_p90",
        "pm10_p90",
        "pm25_alert_days",
        "pm10_alert_days",
        "t_mean_month",
        "rh_mean_month",
        "precip_sum_month",
        "wind_mean_month",
        "month",  # 계절성 표현
        "pm25_mean_lag1m",
        "t_mean_month_lag1m",
        "resp_rate_total_lag1m",
    ]
    target_col = "resp_rate_total"

    # 컬럼별 평균으로 결측치 보정 (주로 pm / 기상 컬럼)
    for col in feature_cols:
        df_model[col] = df_model[col].fillna(df_model[col].mean())

    # -----------------------------
    # 1-4) 학습 / 테스트 분리
    # -----------------------------
    mask_train = df_model["year"] <= 2022
    mask_test = df_model["year"] == 2023

    X_train = df_model.loc[mask_train, feature_cols]
    y_train = df_model.loc[mask_train, target_col]
    X_test = df_model.loc[mask_test, feature_cols]
    y_test = df_model.loc[mask_test, target_col]

    print(f"Train size: {len(X_train)}")
    print(f"Test size : {len(X_test)}")

    # -----------------------------
    # 1-5) 하이퍼파라미터 탐색
    # -----------------------------
    max_depth_list = [3, 4, 5]
    min_samples_leaf_list = [10, 20]

    results = []

    for max_depth in max_depth_list:
        for min_leaf in min_samples_leaf_list:
            model = DecisionTreeRegressor(
                random_state=42,
                max_depth=max_depth,
                min_samples_leaf=min_leaf,
            )
            model.fit(X_train, y_train)

            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            mae_train = mean_absolute_error(y_train, y_pred_train)
            rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
            mae_test = mean_absolute_error(y_test, y_pred_test)
            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

            results.append(
                {
                    "max_depth": max_depth,
                    "min_samples_leaf": min_leaf,
                    "MAE_train": mae_train,
                    "RMSE_train": rmse_train,
                    "MAE_test": mae_test,
                    "RMSE_test": rmse_test,
                }
            )

    results_df = pd.DataFrame(results).sort_values("RMSE_test").reset_index(drop=True)
    print("\n=== 하이퍼파라미터 탐색 결과 (상위 몇 개) ===")
    print(results_df.head())

    # CSV로 저장
    results_df.to_csv(dt_results_out, index=False, encoding="utf-8-sig")
    print("하이퍼파라미터 결과 저장:", dt_results_out)

    # RMSE_test 최소 조합을 최종 모델 파라미터로 사용
    best = results_df.iloc[0]
    best_depth = int(best["max_depth"])
    best_leaf = int(best["min_samples_leaf"])

    print("\n=== 선택된 최종 파라미터 ===")
    print(f"max_depth={best_depth}, min_samples_leaf={best_leaf}")

    # -----------------------------
    # 1-6) 최종 모델 (2018~2023 전체 학습)
    # -----------------------------
    mask_train_all = df_model["year"] <= 2023
    X_train_all = df_model.loc[mask_train_all, feature_cols]
    y_train_all = df_model.loc[mask_train_all, target_col]

    final_model = DecisionTreeRegressor(
        random_state=42,
        max_depth=best_depth,
        min_samples_leaf=best_leaf,
    )
    final_model.fit(X_train_all, y_train_all)

    # 전체 행(2018-02 이후)에 대해 예측
    df_model["y_hat"] = final_model.predict(df_model[feature_cols])

    # 오차 계산
    df_model["error_abs"] = (df_model["resp_rate_total"] - df_model["y_hat"]).abs()
    df_model["error_pct"] = (
        df_model["error_abs"] / df_model["resp_rate_total"].replace(0, np.nan) * 100.0
    )

    # 월 단위 예측 vs 실제 CSV (맵/검증에서 사용)
    out_cols = [
        "year",
        "month",
        "ym",
        "sido_name",
        "resp_rate_total",
        "y_hat",
        "error_abs",
        "error_pct",
    ]
    df_out = df_model[out_cols].copy()
    df_out.to_csv(monthly_pred_out, index=False, encoding="utf-8-sig")
    print("월 단위 예측 결과 저장:", monthly_pred_out)

    # 시도별 평균 오차 요약
    error_summary = (
        df_out.groupby("sido_name")["error_pct"]
        .agg(["count", "mean", "min", "max"])
        .reset_index()
        .rename(
            columns={
                "count": "n_months",
                "mean": "mean_error_pct",
                "min": "min_error_pct",
                "max": "max_error_pct",
            }
        )
    )
    error_summary.to_csv(error_summary_out, index=False, encoding="utf-8-sig")
    print("시도별 오차 요약 저장:", error_summary_out)

    # -----------------------------
    # 1-7) 선형 그래프 & 트리 시각화 저장
    # -----------------------------
    pred_plot_path = OUTPUT_DIR / "pred_vs_actual.png"
    tree_plot_path = OUTPUT_DIR / "decision_tree_full.png"

    # (1) 실제 vs 예측 라인 플롯
    plt.figure(figsize=(10, 5))
    plt.plot(df_out["resp_rate_total"].values, label="Actual")
    plt.plot(df_out["y_hat"].values, label="Predicted")
    plt.title("Prediction vs Actual (monthly, all regions)")
    plt.xlabel("Index (sorted by region/year/month)")
    plt.ylabel("Respiratory rate per 100k")
    plt.legend()
    plt.tight_layout()
    plt.savefig(pred_plot_path, dpi=150)
    plt.close()
    print("실제 vs 예측 그래프 저장:", pred_plot_path)

    # (2) 결정트리 구조
    plt.figure(figsize=(18, 10))
    plot_tree(
        final_model,
        feature_names=feature_cols,
        filled=True,
        rounded=True,
        fontsize=6,
    )
    plt.title("Decision Tree Visualization (Monthly Regression)")
    plt.tight_layout()
    plt.savefig(tree_plot_path, dpi=150)
    plt.close()
    print("결정트리 시각화 저장:", tree_plot_path)

    return df_out


# ============================================
# 2. 월 예측값 기반 일 단위 위험지수 계산
# ============================================

SEASON_WEIGHTS = {
    "winter": 1.20,  # 12, 1, 2
    "spring": 1.00,  # 3, 4, 5
    "summer": 0.90,  # 6, 7, 8
    "autumn": 0.95,  # 9, 10, 11
}


def month_to_season(m: int) -> str:
    if m in (12, 1, 2):
        return "winter"
    if m in (3, 4, 5):
        return "spring"
    if m in (6, 7, 8):
        return "summer"
    return "autumn"


def load_monthly_predictions(path: Path = MONTHLY_PRED_PATH) -> pd.DataFrame:
    """
    monthly_predict_vs_actual.csv 로드
    - 최소한 year, month, sido_name, y_hat 컬럼이 있어야 함.
    - ym 컬럼이 있으면 그대로 유지한다.
    """
    df = pd.read_csv(path)
    if "y_hat" not in df.columns:
        raise ValueError("monthly CSV에 'y_hat' 컬럼이 필요합니다.")
    if "year" not in df.columns or "month" not in df.columns:
        if "ym" in df.columns:
            df["year"] = df["ym"].str.slice(0, 4).astype(int)
            df["month"] = df["ym"].str.slice(5, 7).astype(int)
        else:
            raise ValueError(
                "monthly CSV에는 'year', 'month' 또는 'ym' 컬럼이 필요합니다."
            )
    if "sido_name" not in df.columns:
        raise ValueError("monthly CSV에는 'sido_name' 컬럼이 필요합니다.")
    return df


def get_daily_risk(
    date_str: str,
    monthly_preds_df: Optional[pd.DataFrame] = None,
    env_obs_df: Optional[pd.DataFrame] = None,
    ref_df: Optional[pd.DataFrame] = None,
    percentile_clip: Tuple[int, int] = (10, 90),
    output_csv_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    월 단위 예측값(y_hat)을 이용해 특정 날짜에 대한
    일 단위 예상 환자율 + 위험 점수/등급을 근사 계산한다.

    Parameters
    ----------
    date_str : 'YYYY-MM-DD'
    monthly_preds_df : 미리 로드한 monthly_predict_vs_actual DataFrame (없으면 자동 로드)
    env_obs_df : 선택 입력. 시도별 오늘 대기질/기상 관측값
        - 컬럼 예시: ['sido_name', 'pm25', 'pm10', 't', 'rh']
    ref_df : 선택 입력. 시도×월별 과거 평균 환경값 테이블
        - 컬럼 예시: ['sido_name','month','pm25_month_mean','pm10_month_mean','t_month_mean','rh_month_mean']
    percentile_clip : 위험 점수 스케일링에 사용할 하위/상위 분위수
    output_csv_path : 결과 CSV 저장 경로 (None이면 저장하지 않음)
    """
    # 월 예측값 로드
    if monthly_preds_df is None:
        monthly_preds_df = load_monthly_predictions()

    # 날짜 파싱
    date = datetime.strptime(date_str, "%Y-%m-%d")
    year = date.year
    month = date.month
    days_in_month = calendar.monthrange(year, month)[1]
    season = month_to_season(month)
    default_weight = SEASON_WEIGHTS[season]

    preds = monthly_preds_df.copy()

    # 요청 연월에 해당하는 행만 선택 (없으면 바로 이전 달로 fallback)
    preds_now = preds[(preds["year"] == year) & (preds["month"] == month)].copy()
    if preds_now.empty:
        prev_month = month - 1 if month > 1 else 12
        prev_year = year if month > 1 else year - 1
        preds_now = preds[
            (preds["year"] == prev_year) & (preds["month"] == prev_month)
        ].copy()
        if preds_now.empty:
            raise ValueError(
                f"{year}-{month:02d} 및 이전 달에 대한 월 예측값이 없습니다."
            )

    # 월 예측값 → 기본 일 단위 값
    preds_now["base_daily"] = preds_now["y_hat"] / days_in_month

    # 시도별 weight 계산
    weights = []
    for _, row in preds_now.iterrows():
        sido = row["sido_name"]
        w = default_weight

        if env_obs_df is not None and sido in env_obs_df["sido_name"].values:
            # 관측 환경값 사용
            env_row = env_obs_df[env_obs_df["sido_name"] == sido].iloc[0].to_dict()
            if ref_df is not None and sido in ref_df["sido_name"].values:
                ref_row = (
                    ref_df[(ref_df["sido_name"] == sido) & (ref_df["month"] == month)]
                    .iloc[0]
                    .to_dict()
                )
                pm25_ref = ref_row.get(
                    "pm25_month_mean", max(env_row.get("pm25", 1), 1)
                )
                pm10_ref = ref_row.get(
                    "pm10_month_mean", max(env_row.get("pm10", 1), 1)
                )
                pm25_ratio = env_row.get("pm25", pm25_ref) / max(pm25_ref, 1e-6)
                pm10_ratio = env_row.get("pm10", pm10_ref) / max(pm10_ref, 1e-6)
                temp_ref = ref_row.get("t_month_mean", env_row.get("t", 20))
                temp_diff = temp_ref - env_row.get("t", temp_ref)
                temp_factor = 1.0 + 0.02 * temp_diff
                w_raw = 0.4 * pm25_ratio + 0.3 * pm10_ratio + 0.25 * temp_factor + 0.05
                w = float(np.power(w_raw, 0.25))
            else:
                pm25 = float(env_row.get("pm25", 0.0))
                if pm25 <= 25:
                    w = 0.9
                elif pm25 <= 50:
                    w = 1.05
                elif pm25 <= 75:
                    w = 1.2
                else:
                    w = 1.4

        # 안전 범위 클리핑
        w = max(0.7, min(1.6, w))
        weights.append(w)

    preds_now["w_today"] = weights
    preds_now["y_day_hat"] = preds_now["base_daily"] * preds_now["w_today"]

    # 전체 월 예측값 기준으로 일 단위 분포 추정 → 위험 점수 스케일링 범위 계산
    days_all = preds.apply(
        lambda r: calendar.monthrange(int(r["year"]), int(r["month"]))[1], axis=1
    )
    base_daily_all = preds["y_hat"] / days_all

    low_q, high_q = percentile_clip
    p_low = float(np.nanpercentile(base_daily_all, low_q))
    p_high = float(np.nanpercentile(base_daily_all, high_q))
    if p_high - p_low < 1e-6:
        p_low = float(np.nanmin(base_daily_all))
        p_high = float(np.nanmax(base_daily_all) + 1.0)

    def map_to_score(x: float) -> float:
        raw = (x - p_low) / (p_high - p_low) * 100.0
        return float(max(0.0, min(100.0, raw)))

    preds_now["risk_score"] = preds_now["y_day_hat"].apply(map_to_score)

    def score_to_level(s: float) -> str:
        if s <= 25.0:
            return "Low"
        if s <= 50.0:
            return "Moderate"
        if s <= 75.0:
            return "High"
        return "Very High"

    preds_now["risk_level"] = preds_now["risk_score"].apply(score_to_level)

    preds_now["date"] = date_str

    out_cols = [
        "date",
        "year",
        "month",
        "ym",
        "sido_name",
        "y_hat",
        "base_daily",
        "w_today",
        "y_day_hat",
        "risk_score",
        "risk_level",
    ]
    out = preds_now[out_cols].copy()

    if output_csv_path is not None:
        out.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
        print("일 단위 위험지수 CSV 저장:", output_csv_path)

    return out


# ============================================
# 3. 스크립트 직접 실행 시: 전체 파이프라인 수행
# ============================================

if __name__ == "__main__":
    print(">>> 월 단위 회귀트리 학습 및 평가 시작")
    monthly_df = train_monthly_tree()

    print("\n>>> 예시 날짜(2023-12-01)에 대한 일 단위 위험지수 계산")
    daily_out_path = OUTPUT_DIR / "daily_risk_2023-12-01.csv"
    daily_df = get_daily_risk(
        "2023-12-01",
        monthly_preds_df=monthly_df,
        output_csv_path=daily_out_path,
    )

    print("\n파이프라인 완료.")
    print(daily_df.head())
