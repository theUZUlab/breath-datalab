"""
HIRA 4단질병 엑셀(J00~J99) → 시·도 × 연월별 호흡기 환자 합계 매핑 스크립트

- 입력:
    data/raw/health/hira/hira_4th_disease/J00_201801_202012.xlsx 등
- 처리:
    · 각 엑셀에서 '요양기관소재지구분' 별 월별 '환자수' 컬럼 추출
    · 전국 합계('계') 제거
    · J00~J99 전체 상병코드에 대해 환자수를 합산(resp_total)
- 출력:
    data/processed/health/health_J00_J99_monthly.csv
"""

import os
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# ─────────────────────────────────────────
# 0. 환경 변수 로드
# ─────────────────────────────────────────
load_dotenv()

# ─────────────────────────────────────────
# 1. 경로 설정
#    - RAW_DIR : data/raw/health/hira/{HIRA_OUTPUT_DIR_NAME}
#    - OUT_DIR : data/processed/health
# ─────────────────────────────────────────
# etl/data_mapping/hira/hira_mapping.py 기준으로
# repo 루트는 parents[3]에 위치한다고 가정
ROOT_DIR = Path(__file__).resolve().parents[3]

HIRA_OUTPUT_DIR_NAME = os.getenv("HIRA_OUTPUT_DIR", "hira_4th_disease")

# 원본 엑셀 위치: data/raw/health/hira/hira_4th_disease
#    (한국어: "건강/HIRA 4단 질병 원시 엑셀 데이터 저장 폴더")
RAW_DIR = ROOT_DIR / "data" / "raw" / "health" / "hira" / HIRA_OUTPUT_DIR_NAME

# 가공 결과(csv) 저장 위치: data/processed/health
#    (한국어: "건강 관련 전처리·집계 결과 저장 폴더")
OUT_DIR = ROOT_DIR / "data" / "processed" / "health"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1-1. 엑셀 파일들이 있는 폴더
DATA_DIR = RAW_DIR

# 1-2. J00~J99 엑셀 파일 전부 찾기
files = sorted(DATA_DIR.glob("J*.xlsx"))


# ─────────────────────────────────────────
# 2. 헬퍼 함수: 월 라벨 정제
#    '2018년 01월' → '2018-01'
# ─────────────────────────────────────────
def clean_month_label(s: str) -> str:
    """
    '2018년 01월' 같은 문자열을 '2018-01' 형식으로 변환한다.
    매칭이 안 되면 원본 문자열을 그대로 반환한다.
    """
    s = str(s)
    m = re.search(r"(\d{4})\D+(\d{1,2})", s)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}"
    return s


# ─────────────────────────────────────────
# 3. 각 엑셀(J코드별) → Long 포맷으로 변환
# ─────────────────────────────────────────
all_long: list[pd.DataFrame] = []

for path in files:
    diag_code = path.stem.split("_")[0]  # 예: 'J00_201801_202012' → 'J00'
    print(f"[처리 중] {path.name} ({diag_code})")

    # 3-1. 엑셀 읽기
    #      3번째 줄(인덱스 2), 4번째 줄(인덱스 3)을 MultiIndex 헤더로 사용
    df_raw = pd.read_excel(path, header=[2, 3])
    df_raw.columns = pd.MultiIndex.from_tuples(df_raw.columns)

    # 3-2. '항목', '요양기관소재지구분' 컬럼 찾기 (두 번째 레벨 기준)
    diag_col = None  # 항목
    sido_col = None  # 요양기관소재지구분

    for col in df_raw.columns:
        if isinstance(col, tuple) and len(col) > 1:
            if col[1] == "항목":
                diag_col = col
            elif col[1] == "요양기관소재지구분":
                sido_col = col

    if diag_col is None or sido_col is None:
        print(
            "  !! '항목' / '요양기관소재지구분' 컬럼을 찾지 못했음. 컬럼 구조 확인 필요."
        )
        print("  현재 컬럼 예시:", df_raw.columns.tolist()[:10])
        continue

    # 3-3. ID 부분 추출
    id_df = df_raw[[diag_col, sido_col]].copy()
    id_df.columns = ["diag_raw", "sido_name"]

    # 전국 합계 '계' 제거
    mask = id_df["sido_name"] != "계"
    id_df = id_df[mask]
    df_sub = df_raw.loc[mask].copy()

    # 3-4. 값 컬럼 중 '환자수'만 선택
    patient_cols = [
        c
        for c in df_sub.columns
        if isinstance(c, tuple) and len(c) > 1 and c[1] == "환자수"
    ]
    patients = df_sub[patient_cols].copy()

    # ('2018년 01월','환자수') → '2018-01'
    month_labels = [clean_month_label(c[0]) for c in patient_cols]
    patients.columns = month_labels

    # 시도 이름 붙이기
    patients["sido_name"] = id_df["sido_name"].values

    # 3-5. wide → long (시도 × 연월별 환자수)
    long = patients.melt(
        id_vars=["sido_name"],
        var_name="ym",
        value_name="patient_cnt",
    )

    # J00~J99 상병코드 보존 (필요 시 필터링/디버깅에 사용 가능)
    long["diag_code"] = diag_code
    all_long.append(long)

# ─────────────────────────────────────────
# 4. 전체 J코드 데이터 결합 및 숫자 변환
# ─────────────────────────────────────────
if not all_long:
    raise RuntimeError(f"J코드 엑셀 파일을 찾지 못했음. 경로를 확인하세요: {DATA_DIR}")

health_long = pd.concat(all_long, ignore_index=True)

# '환자수'를 숫자로 강제 변환 (문자/공백/기타값 → NaN → 0)
health_long["patient_cnt"] = (
    pd.to_numeric(health_long["patient_cnt"], errors="coerce").fillna(0).astype("int64")
)

# ─────────────────────────────────────────
# 5. 시·도 × ym 기준으로 J00~J99 환자수 합계 → resp_total
# ─────────────────────────────────────────
agg = health_long.groupby(["sido_name", "ym"], as_index=False).agg(
    resp_total=("patient_cnt", "sum")
)

# ─────────────────────────────────────────
# 6. year, month 파생 컬럼 생성
# ─────────────────────────────────────────
agg["ym"] = agg["ym"].astype(str)
agg["year"] = agg["ym"].str.slice(0, 4).astype(int)
agg["month"] = agg["ym"].str.slice(5, 7).astype(int)

# 컬럼 순서 정리
agg = agg[
    [
        "year",
        "month",
        "ym",
        "sido_name",
        "resp_total",
    ]
]

# ─────────────────────────────────────────
# 7. CSV 저장
# ─────────────────────────────────────────
out_path = OUT_DIR / "health_J00_J99_monthly.csv"
agg.to_csv(out_path, index=False, encoding="utf-8-sig")

print(f"완료: {out_path}")
print(agg.head())
print(len(agg), "rows 생성됨")
