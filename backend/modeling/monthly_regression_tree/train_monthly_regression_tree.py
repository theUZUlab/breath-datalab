# ----------------------------------------
# 월 단위 회귀트리(DecisionTreeRegressor)로
# 시·도 × 연월별 resp_rate_total 예측 모델을 학습하고,
# monthly_predict_vs_actual.csv를 생성하는 스크립트
# ----------------------------------------

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

# ----------------------------------------
# 0. 경로 설정
#    - 프로젝트 루트 기준: data/ 아래만 사용
# ----------------------------------------
DATA_DIR = Path("data")

INPUT_PATH = DATA_DIR / "mart" / "merged_monthly.csv"

MODEL_DIR = DATA_DIR / "model" / "monthly_regression_tree"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PRED_PATH = MODEL_DIR / "monthly_predict_vs_actual.csv"
OUTPUT_HPARAM_PATH = MODEL_DIR / "monthly_dt_results.csv"

print("입력 파일 경로:", INPUT_PATH.resolve())
print("출력 디렉토리:", MODEL_DIR.resolve())

# ----------------------------------------
# 1. 데이터 로드 및 기본 정보 확인
# ----------------------------------------
df = pd.read_csv(INPUT_PATH)

print("\n=== head ===")
print(df.head())
print("\n=== info ===")
print(df.info())
print("\n=== 결측값 개수 ===")
print(df.isna().sum())
print("\n=== 기술통계 ===")
print(df.describe(include="all"))

# ----------------------------------------
# 2. 분석/모델링에 사용할 컬럼만 선택
# ----------------------------------------
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

# 시차 특징 생성을 위해 시도별/연도/월 정렬
df_model = df_model.sort_values(["sido_name", "year", "month"]).reset_index(drop=True)

# ----------------------------------------
# 3. lag 특징 생성 (이전 달 정보)
#    - pm25_mean_lag1m
#    - t_mean_month_lag1m
#    - resp_rate_total_lag1m
# ----------------------------------------
df_model["pm25_mean_lag1m"] = df_model.groupby("sido_name")["pm25_mean"].shift(1)
df_model["t_mean_month_lag1m"] = df_model.groupby("sido_name")["t_mean_month"].shift(1)
df_model["resp_rate_total_lag1m"] = df_model.groupby("sido_name")[
    "resp_rate_total"
].shift(1)

# lag가 NaN인 첫 달 행 제거 (학습/평가 대상에서 제외)
df_model = df_model.dropna(
    subset=["pm25_mean_lag1m", "t_mean_month_lag1m", "resp_rate_total_lag1m"]
).copy()

# ----------------------------------------
# 4. 특징 컬럼(feature_cols) 및 타깃 설정
#    - 팀 공통 기준으로 사용할 feature 목록
# ----------------------------------------
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
    "month",  # 계절성 표현 (1~12)
    "pm25_mean_lag1m",
    "t_mean_month_lag1m",
    "resp_rate_total_lag1m",
]

target_col = "resp_rate_total"

# 아직 남아있는 결측치(주로 pm/기상 관련 컬럼)를 컬럼별 평균으로 채움
for col in feature_cols:
    df_model[col] = df_model[col].fillna(df_model[col].mean())

# ----------------------------------------
# 5. 학습/테스트 기간 설정
#    - 예시: 2018~2022년 학습, 2023년 테스트
# ----------------------------------------
mask_train = df_model["year"] <= 2022
mask_test = df_model["year"] == 2023

X_train = df_model.loc[mask_train, feature_cols]
y_train = df_model.loc[mask_train, target_col]
X_test = df_model.loc[mask_test, feature_cols]
y_test = df_model.loc[mask_test, target_col]

print("\nTrain size:", len(X_train))
print("Test size :", len(X_test))

# ----------------------------------------
# 6. 하이퍼파라미터 탐색
#    - max_depth × min_samples_leaf 그리드 서치
# ----------------------------------------
max_depth_list = [3, 4, 5, 6]
min_samples_leaf_list = [10, 20, 30]

results = []

for max_depth in max_depth_list:
    for min_samples_leaf in min_samples_leaf_list:
        model = DecisionTreeRegressor(
            random_state=42,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
        )
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        mae_train = mean_absolute_error(y_train, y_pred_train)
        # squared=False는 버전 이슈가 있어서 직접 sqrt로 RMSE 계산
        rmse_train = sqrt(mean_squared_error(y_train, y_pred_train))

        mae_test = mean_absolute_error(y_test, y_pred_test)
        rmse_test = sqrt(mean_squared_error(y_test, y_pred_test))

        results.append(
            {
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf,
                "MAE_train": mae_train,
                "RMSE_train": rmse_train,
                "MAE_test": mae_test,
                "RMSE_test": rmse_test,
            }
        )

results_df = pd.DataFrame(results).sort_values("RMSE_test")
print("\n=== 하이퍼파라미터 결과 (RMSE_test 오름차순) ===")
print(results_df)

# 결과를 CSV로도 저장 (선택이지만 나중에 분석/보고서에 사용하기 편함)
results_df.to_csv(OUTPUT_HPARAM_PATH, index=False, encoding="utf-8-sig")
print("\n하이퍼파라미터 결과 저장 완료:", OUTPUT_HPARAM_PATH.resolve())

# ----------------------------------------
# 7. 최종 파라미터 선택 및 전체(2018~2023) 재학습
#    - 2024년은 완전 미래 데이터처럼 예측만 수행
# ----------------------------------------
best = results_df.iloc[0]
best_max_depth = int(best["max_depth"])
best_min_samples_leaf = int(best["min_samples_leaf"])

print("\n=== 선택된 최종 파라미터 ===")
print(best)

mask_train_all = df_model["year"] <= 2023
X_train_all = df_model.loc[mask_train_all, feature_cols]
y_train_all = df_model.loc[mask_train_all, target_col]

final_model = DecisionTreeRegressor(
    random_state=42,
    max_depth=best_max_depth,
    min_samples_leaf=best_min_samples_leaf,
)
final_model.fit(X_train_all, y_train_all)

# ----------------------------------------
# 8. 전체 기간(2018-02 이후 lag 있는 구간)에 대한 예측 및 오차 계산
# ----------------------------------------
X_all = df_model[feature_cols]
df_model["y_hat"] = final_model.predict(X_all)

df_model["error_abs"] = (df_model[target_col] - df_model["y_hat"]).abs()
df_model["error_pct"] = (
    df_model["error_abs"] / df_model[target_col].replace(0, np.nan) * 100
)

# ----------------------------------------
# 9. 최종 출력 테이블 정리 및 CSV 저장
#    - A, C, 웹/노트북에서 공통으로 사용할 월 단위 성능/예측 테이블
# ----------------------------------------
out_cols = [
    "year",
    "month",
    "ym",
    "sido_name",
    "resp_rate_total",  # 실제값
    "y_hat",  # 예측값
    "error_abs",
    "error_pct",
]

df_out = df_model[out_cols].copy()
df_out.to_csv(OUTPUT_PRED_PATH, index=False, encoding="utf-8-sig")

print("\nmonthly_predict_vs_actual.csv 저장 완료!")
print("경로:", OUTPUT_PRED_PATH.resolve())
print(df_out.head())
