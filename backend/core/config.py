from pathlib import Path


# ─────────────────────────────────────
# 1) 프로젝트 루트 / 백엔드 / 데이터 경로
# ─────────────────────────────────────

#   parents[0] → PROJECT_ROOT/backend/core
#   parents[1] → PROJECT_ROOT/backend
#   parents[2] → PROJECT_ROOT
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
BACKEND_DIR: Path = PROJECT_ROOT / "backend"
DATA_DIR: Path = BACKEND_DIR / "data"

# 세부 데이터 디렉터리
MART_DIR: Path = DATA_DIR / "mart"
PROCESSED_DIR: Path = DATA_DIR / "processed"
MODEL_DIR: Path = DATA_DIR / "model"


# ─────────────────────────────────────
# 2) 자주 쓰는 파일 경로 상수
# ─────────────────────────────────────

# B가 만든 병합 월 데이터 (월 회귀트리 학습에 사용)
MERGED_MONTHLY_PATH: Path = MART_DIR / "merged_monthly.csv"

# 월 단위 예측 vs 실제 결과 (맵/탭1용)
MONTHLY_PREDICT_VS_ACTUAL_PATH: Path = MART_DIR / "monthly_predict_vs_actual.csv"

# 월 단위 환경값 (data/processed 내 파일들)
# - 시도×월별 PM2.5/PM10 관련 요약
AIRKOREA_PM_MONTHLY_PATH: Path = PROCESSED_DIR / "airkorea_pm25_pm10_monthly.csv"

# - 시도×월별 기상(ASOS) 월 평균
ASOS_MONTHLY_PATH: Path = PROCESSED_DIR / "asos_2018_2024_monthly.csv"

# (선택) 일 단위 위험지수 실험 등 모델 관련 아웃풋 디렉터리
DAILY_RISK_MODEL_DIR: Path = MODEL_DIR / "daily_risk_index"


# ─────────────────────────────────────
# 3) 유틸 함수 (선택 사항)
# ─────────────────────────────────────


def ensure_data_paths() -> None:
    """
    서버 기동 전에 기본 디렉터리가 존재하는지 확인하는 함수.
    실제 서비스에서 필요 없으면 호출 안 해도 됨.
    """
    for path in [DATA_DIR, MART_DIR, PROCESSED_DIR, MODEL_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def debug_print_paths() -> None:
    """
    디버깅용: uvicorn 실행 전에 경로가 제대로 잡혔는지 확인하고 싶을 때 사용.
    """
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("BACKEND_DIR :", BACKEND_DIR)
    print("DATA_DIR    :", DATA_DIR)
    print("MART_DIR    :", MART_DIR)
    print("PROCESSED_DIR:", PROCESSED_DIR)
    print("MODEL_DIR   :", MODEL_DIR)
    print("MERGED_MONTHLY_PATH          :", MERGED_MONTHLY_PATH)
    print("MONTHLY_PREDICT_VS_ACTUAL_PATH:", MONTHLY_PREDICT_VS_ACTUAL_PATH)
    print("AIRKOREA_PM_MONTHLY_PATH     :", AIRKOREA_PM_MONTHLY_PATH)
    print("ASOS_MONTHLY_PATH            :", ASOS_MONTHLY_PATH)
    print("DAILY_RISK_MODEL_DIR         :", DAILY_RISK_MODEL_DIR)
