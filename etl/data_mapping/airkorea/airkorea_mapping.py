"""
AirKorea PM2.5 / PM10 월·시도 매핑 스크립트

1) 입력 파일 구조
   - PM2.5 (초미세먼지) 일평균 엑셀 (pmRelay → 엑셀 다운로드 결과)
     경로: backend/data/climate/raw/airkorea/pm25/
     파일명 예: airkorea_pm25_강원_201801.xls

   - PM10 (미세먼지) 일평균 엑셀
     경로: backend/data/climate/raw/airkorea/pm10/
     파일명 예: airkorea_pm10_강원_201801.xls

   엑셀 구조 예시:
     (위쪽 몇 줄은 제목/공백)
     측정망 | 측정소명 | 1일 | 2일 | ... | 31일
     교외대기 | [강원 고성군]간성읍 | ... (일별 값)
     ...

     - 한 행: 하나의 측정소
     - '1일'~'31일' 컬럼: 해당 날짜의 일평균 농도(µg/m3)

2) 집계 로직
   - 파일명에서 시도명 / 연월(YYYYMM)을 파싱
       airkorea_pm25_강원_201801.xls
       → sido_name='강원', ym='2018-01', year=2018, month=1

   - 엑셀은 header=None 으로 통째로 읽은 뒤,
     위쪽 몇 줄에서 '측정소명' + '1일/2일...' 이 있는 행을 헤더로 설정

   - 이후 '1일'~'31일' 컬럼만 사용
     · 각 'n일' 컬럼: 모든 측정소의 값 평균 → 시도 일평균
     · day_means = [일1 평균, 일2 평균, ..., 일N 평균]

   - 월 단위 지표 (PM2.5 / PM10 각각)
     · pm*_mean        : day_means 의 평균
     · pm*_p90         : day_means 의 90퍼센타일
     · pm*_alert_days  : day_means >= 기준값 인 날짜 개수
                          (기본값 PM2.5=35, PM10=100, .env에서 변경 가능)

   - 비어있는 값, NaN, -999 등은 전부 0으로 간주

3) 출력
   - 경로: data/processed/climate/airkorea_pm25_pm10_monthly.csv
   - 컬럼:
       year
       month
       ym            (YYYY-MM)
       sido_name
       pm25_mean
       pm10_mean
       pm25_p90
       pm10_p90
       pm25_alert_days
       pm10_alert_days
"""

import sys
from pathlib import Path
import os

import pandas as pd
from dotenv import load_dotenv


# ─────────────────────────────────────
# 0. 환경 변수 로드
# ─────────────────────────────────────
load_dotenv()


# ─────────────────────────────────────
# 1. 경로 설정
#    - ROOT_DIR : 레포 루트
#    - RAW_PM25_DIR: data/raw/climate/airkorea/pm25
#    - RAW_PM10_DIR: data/raw/climate/airkorea/pm10
#    - OUT_DIR: data/processed/climate
# ─────────────────────────────────────

from pathlib import Path
import sys
import os

ROOT_DIR = Path(__file__).resolve().parents[3]

# data 루트
DATA_ROOT = ROOT_DIR / "data"

# 원본(xls) 위치
RAW_PM25_DIR = DATA_ROOT / "raw" / "climate" / "airkorea" / "pm25"
RAW_PM10_DIR = DATA_ROOT / "raw" / "climate" / "airkorea" / "pm10"

# 결과(csv) 저장 위치
OUT_DIR = DATA_ROOT / "processed" / "climate"
OUT_DIR.mkdir(parents=True, exist_ok=True)

if not RAW_PM25_DIR.exists():
    print(f"[에러] PM2.5 디렉터리 없음: {RAW_PM25_DIR}", file=sys.stderr)
    sys.exit(1)

if not RAW_PM10_DIR.exists():
    print(f"[에러] PM10 디렉터리 없음: {RAW_PM10_DIR}", file=sys.stderr)
    sys.exit(1)


# ─────────────────────────────────────
# 2. 경보 기준값 설정
#    - 기본값:
#        PM2.5: 35.0
#        PM10 : 100.0
#    - .env 에서 override 가능
#      AIRKOREA_PM25_ALERT_THRESHOLD
#      AIRKOREA_PM10_ALERT_THRESHOLD
# ─────────────────────────────────────


def _get_float_env(key: str, default: float) -> float:
    val = os.getenv(key, str(default))
    try:
        return float(val)
    except ValueError:
        return default


PM25_ALERT_THRESHOLD = _get_float_env("AIRKOREA_PM25_ALERT_THRESHOLD", 35.0)
PM10_ALERT_THRESHOLD = _get_float_env("AIRKOREA_PM10_ALERT_THRESHOLD", 100.0)

print(f"[INFO] PM2.5 alert threshold = {PM25_ALERT_THRESHOLD}")
print(f"[INFO] PM10  alert threshold = {PM10_ALERT_THRESHOLD}")


# ─────────────────────────────────────
# 3. 파일명 → 시도명, 연월 파싱
#    - airkorea_pm25_강원_201801.xls
#      → sido_name='강원', ym='2018-01', year=2018, month=1
# ─────────────────────────────────────


def parse_sido_ym_from_name(path: Path, pollutant_tag: str):
    """
    파일명에서 시도명과 연월(YYYY-MM)을 파싱한다.
    예: airkorea_pm25_강원_201801.xls
    """
    stem = path.stem  # 'airkorea_pm25_강원_201801'
    parts = stem.split("_")

    # ['airkorea', 'pm25', '강원', '201801'] 형식 가정
    try:
        idx = parts.index(pollutant_tag)
    except ValueError:
        raise ValueError(f"파일명에 '{pollutant_tag}' 토큰이 없음: {stem}")

    try:
        sido_name = parts[idx + 1]
        yyyymm = parts[idx + 2]
    except IndexError:
        raise ValueError(f"파일명에서 시도/연월 파싱 실패: {stem}")

    if len(yyyymm) != 6 or not yyyymm.isdigit():
        raise ValueError(f"연월 형식 이상(YYYYMM 아님): {yyyymm} (파일: {stem})")

    year = int(yyyymm[:4])
    month = int(yyyymm[4:6])
    ym = f"{year:04d}-{month:02d}"

    return sido_name, ym, year, month


# ─────────────────────────────────────
# 4. 엑셀 로더 (header=None → 헤더는 나중에 찾음)
# ─────────────────────────────────────


def load_excel_as_dataframe(path: Path) -> pd.DataFrame:
    """
    xls 파일을 DataFrame으로 읽는다.
    - 항상 header=None 으로 읽어서, 헤더 행은 나중에 직접 찾는다.
    - pandas 기본 엔진 시도 후, 실패하면 engine='xlrd'로 재시도.
    """
    try:
        return pd.read_excel(path, header=None)
    except Exception as e1:
        try:
            return pd.read_excel(path, header=None, engine="xlrd")
        except Exception as e2:
            raise RuntimeError(
                f"엑셀 읽기 실패: {path}\n"
                f"  1차 오류: {e1}\n"
                f"  2차 오류: {e2}\n"
                f"  → xls 파일이면 'pip install xlrd>=2.0.1' 했는지 확인 필요"
            )


# ─────────────────────────────────────
# 5. 엑셀에서 '1일'~'31일' 컬럼만 골라서 요약
#    - 비어있는 값/NaN/-999 는 모두 0으로 간주
#    - 헤더 행은 '측정소명' + '1일/2일...' 이 있는 줄을 자동 탐색
# ─────────────────────────────────────

DAY_LABELS = [f"{d}일" for d in range(1, 32)]  # '1일' ~ '31일'


def summarize_file(
    path: Path,
    pollutant_tag: str,
    alert_threshold: float,
    col_prefix: str,
) -> pd.DataFrame:
    """
    단일 엑셀 파일을 읽어 (시도, 연월)에 대한 월 집계 1행 DataFrame 생성.

    - 파일명에서 year/month/ym/sido_name 파싱
    - 엑셀은 header=None 으로 읽어온 뒤,
      위쪽 몇 줄에서 '측정소명' + '1일/2일...' 이 있는 행을 헤더로 재설정
    - 그 뒤 '1일'~'31일' 열만 사용해서 월 지표 계산
    """
    # 1) 파일명에서 메타 정보
    sido_name, ym, year, month = parse_sido_ym_from_name(path, pollutant_tag)

    # 2) 엑셀 로드 (header=None)
    df_raw = load_excel_as_dataframe(path)

    # 완전히 빈 행/열 제거
    df_raw = df_raw.dropna(how="all").dropna(axis=1, how="all")
    if df_raw.empty:
        raise ValueError(f"데이터가 비어 있음: {path.name}")

    # 3) 헤더 행 찾기: 위쪽 10줄 안에서 '측정소명'과 '1일/2일...'이 같이 있는 줄
    header_idx = None
    max_scan = min(10, len(df_raw))
    for i in range(max_scan):
        row_vals = [str(v).strip() for v in df_raw.iloc[i].tolist()]
        if "측정소명" in row_vals and any(lbl in row_vals for lbl in DAY_LABELS):
            header_idx = i
            break

    if header_idx is None:
        raise ValueError(f"헤더 행(측정소명 + 1일~31일)이 없음: {path.name}")

    # 4) 헤더 행을 컬럼으로 사용하고, 그 아래부터 데이터로 사용
    cols = [str(v).strip() for v in df_raw.iloc[header_idx].tolist()]
    df = df_raw.iloc[header_idx + 1 :].copy()
    df.columns = cols

    # 다시 한 번 완전 빈 행 제거
    df = df.dropna(how="all")
    if df.empty:
        raise ValueError(f"헤더 설정 후 데이터가 비어 있음: {path.name}")

    # 5) 1일~31일 컬럼 찾기
    day_cols = []
    for col in df.columns:
        col_str = str(col).strip()

        # '1일' ~ '31일'
        if col_str in DAY_LABELS:
            day_cols.append(col)
            continue

        # 혹시 '1', '2' 처럼 숫자만 있는 경우도 허용
        if col_str.isdigit():
            d = int(col_str)
            if 1 <= d <= 31:
                day_cols.append(col)

    if not day_cols:
        raise ValueError(f"일자 컬럼(1일~31일)이 없음: {path.name}")

    # 6) 날짜별(열 기준) 평균 계산
    day_means = []
    for col in day_cols:
        s = pd.to_numeric(df[col], errors="coerce")

        # 결측/코드값 처리: -999, NaN 등은 0으로 간주
        s = s.replace(-999, 0).fillna(0)

        day_mean = float(s.mean())
        day_means.append(day_mean)

    if not day_means:
        raise ValueError(f"day_means 가 비어 있음: {path.name}")

    day_means = pd.Series(day_means, dtype="float")

    # 7) 월 지표 계산
    pm_mean = float(day_means.mean())
    pm_p90 = float(day_means.quantile(0.9))
    alert_days = int((day_means >= alert_threshold).sum())

    data = {
        "year": year,
        "month": month,
        "ym": ym,
        "sido_name": sido_name,
        f"{col_prefix}_mean": pm_mean,
        f"{col_prefix}_p90": pm_p90,
        f"{col_prefix}_alert_days": alert_days,
    }

    return pd.DataFrame([data])


# ─────────────────────────────────────
# 6. PM2.5 / PM10 각각 월·시도 집계 생성
# ─────────────────────────────────────


def build_pm25_monthly() -> pd.DataFrame:
    files = sorted(RAW_PM25_DIR.glob("airkorea_pm25_*_*.xls"))
    if not files:
        print(f"[에러] PM2.5 파일 없음: {RAW_PM25_DIR}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for p in files:
        try:
            df_one = summarize_file(
                path=p,
                pollutant_tag="pm25",
                alert_threshold=PM25_ALERT_THRESHOLD,
                col_prefix="pm25",
            )
            rows.append(df_one)
            print(f"[PM2.5 OK] {p.name}")
        except Exception as e:
            print(f"[PM2.5 실패] {p.name} → {e}", file=sys.stderr)

    if not rows:
        raise RuntimeError("PM2.5 요약 결과가 비어 있음")

    df_all = pd.concat(rows, ignore_index=True)
    df_all = df_all.sort_values(["sido_name", "year", "month"]).reset_index(drop=True)
    return df_all


def build_pm10_monthly() -> pd.DataFrame:
    files = sorted(RAW_PM10_DIR.glob("airkorea_pm10_*_*.xls"))
    if not files:
        print(f"[에러] PM10 파일 없음: {RAW_PM10_DIR}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for p in files:
        try:
            df_one = summarize_file(
                path=p,
                pollutant_tag="pm10",
                alert_threshold=PM10_ALERT_THRESHOLD,
                col_prefix="pm10",
            )
            rows.append(df_one)
            print(f"[PM10 OK] {p.name}")
        except Exception as e:
            print(f"[PM10 실패] {p.name} → {e}", file=sys.stderr)

    if not rows:
        raise RuntimeError("PM10 요약 결과가 비어 있음")

    df_all = pd.concat(rows, ignore_index=True)
    df_all = df_all.sort_values(["sido_name", "year", "month"]).reset_index(drop=True)
    return df_all


# ─────────────────────────────────────
# 7. PM2.5 / PM10 머지 후 CSV 저장
# ─────────────────────────────────────


def main():
    df_pm25 = build_pm25_monthly()
    df_pm10 = build_pm10_monthly()

    key_cols = ["year", "month", "ym", "sido_name"]

    df_merged = (
        pd.merge(
            df_pm25,
            df_pm10,
            on=key_cols,
            how="outer",
        )
        .sort_values(["sido_name", "year", "month"])
        .reset_index(drop=True)
    )

    # 컬럼 순서 정리 (요청 스키마)
    desired_cols = [
        "year",
        "month",
        "ym",
        "sido_name",
        "pm25_mean",
        "pm10_mean",
        "pm25_p90",
        "pm10_p90",
        "pm25_alert_days",
        "pm10_alert_days",
    ]
    final_cols = [c for c in desired_cols if c in df_merged.columns]
    df_merged = df_merged[final_cols]

    out_file = OUT_DIR / "airkorea_pm25_pm10_monthly.csv"
    df_merged.to_csv(out_file, index=False, encoding="utf-8-sig")

    print(f"[완료] 저장: {out_file}")
    print(df_merged.head())


if __name__ == "__main__":
    main()
