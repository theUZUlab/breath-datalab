import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# ─────────────────────────────────────
# 1. 경로 설정
#    - ROOT_DIR: 레포 루트
#    - RAW_DIR : 기상청 ASOS 월자료 원본(xls/xlsx) 위치
#    - OUT_DIR : 전처리 결과(csv) 저장 위치
# ─────────────────────────────────────

ROOT_DIR = Path(__file__).resolve().parents[3]

RAW_DIR = ROOT_DIR / "data" / "raw" / "climate" / "asos"
OUT_DIR = ROOT_DIR / "data" / "processed" / "climate"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# asos_monthly*.xls / .xlsx 자동 탐색
candidates = sorted(
    list(RAW_DIR.glob("asos_monthly*.xls")) + list(RAW_DIR.glob("asos_monthly*.xlsx"))
)
if not candidates:
    print(f"[에러] {RAW_DIR} 아래에 asos_monthly*.xls(x) 파일이 없음", file=sys.stderr)
    sys.exit(1)

raw_path = candidates[0]
print(f"[INFO] 사용 파일: {raw_path}")


# ─────────────────────────────────────
# 2. 지점명 -> 시도명 매핑 테이블
# ─────────────────────────────────────
station_to_sido = {
    # ── 서울/수도권 ─────────────────
    "서울": "서울",
    "수원": "경기",
    "인천": "인천",
    "파주": "경기",
    "동두천": "경기",
    "양평": "경기",
    # ── 강원 ──────────────────────
    "춘천": "강원",
    "원주": "강원",
    "강릉": "강원",
    "속초": "강원",
    "대관령": "강원",
    # ── 경남/부산/울산 ─────────────
    "부산": "부산",
    "울산": "울산",
    "창원": "경남",
    "진주": "경남",
    "통영": "경남",
    "거제": "경남",
    # ── 경북/대구 ─────────────────
    "대구": "대구",
    "포항": "경북",
    "안동": "경북",
    "문경": "경북",
    "울진": "경북",
    # ── 전남/광주/전북 ─────────────
    "광주": "광주",
    "목포": "전남",
    "순천": "전남",
    "여수": "전남",
    "흑산도": "전남",
    "전주": "전북",
    "군산": "전북",
    "부안": "전북",
    # ── 충청/대전/세종 ─────────────
    "대전": "대전",
    "청주": "충북",
    "충주": "충북",
    "제천": "충북",
    "홍성": "충남",
    "서산": "충남",
    "보령": "충남",
    "세종": "세종",
    # ── 제주 ──────────────────────
    "제주": "제주",
    "서귀포": "제주",
}


# ─────────────────────────────────────
# 3. ASOS 원본 읽기 (xls = 가짜 엑셀 대응)
# ─────────────────────────────────────


def load_raw_asos_excel(path: Path) -> pd.DataFrame:
    """
    ASOS 월자료 파일을 DataFrame으로 읽는다.
    - KMA에서 내려받은 .xls가 실제 엑셀이 아니라 CP949 텍스트인 경우가 많아서
      1차로 read_excel 시도 후 실패하면,
      첫 줄을 읽어 구분자(tab/콤마)를 추론한 뒤 read_csv(cp949, sep=...)로 폴백한다.
    """
    suffix = path.suffix.lower()

    # xlsx / xlsm → 진짜 엑셀로 처리
    if suffix in [".xlsx", ".xlsm"]:
        df_raw = pd.read_excel(path, header=None)

    else:
        # .xls 인데 실제로는 텍스트인 경우가 많음
        try:
            df_raw = pd.read_excel(path, header=None)
        except Exception as e1:
            # 텍스트 파일로 간주하고, 첫 줄을 보고 구분자 추론
            try:
                with open(path, "r", encoding="cp949", errors="ignore") as f:
                    first_line = f.readline()
                if "\t" in first_line:
                    sep = "\t"
                elif "," in first_line:
                    sep = ","
                else:
                    sep = None  # pandas 기본 guess

                df_raw = pd.read_csv(path, encoding="cp949", sep=sep)
                print(
                    f"[INFO] read_excel 실패 → read_csv(cp949, sep={repr(sep)})로 대체 로드"
                )
            except Exception as e2:
                raise RuntimeError(
                    f"ASOS 파일 읽기 실패: {path}\n"
                    f"  1차(read_excel) 오류: {e1}\n"
                    f"  2차(read_csv cp949) 오류: {e2}\n"
                )

    # 완전 빈 행/열 제거
    df_raw = df_raw.dropna(how="all").dropna(axis=1, how="all")
    if df_raw.empty:
        raise ValueError(f"데이터가 비어 있음: {path}")

    print(f"[INFO] 원본 shape: {df_raw.shape}")
    return df_raw


def find_header_and_build(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    - 경우1) 이미 컬럼에 '지점', '지점명', '일시/년월' 이 있으면
            → 첫 행이 헤더라고 보고 그대로 사용
    - 경우2) header=None 으로 읽은 경우
            → 위쪽 몇 줄에서 헤더 행을 찾아서 설정
    """
    cols_str = [str(c).strip() for c in df_raw.columns]

    has_station_col = ("지점" in cols_str) or any("지점번호" in x for x in cols_str)
    has_name_col = "지점명" in cols_str
    has_time_col = ("일시" in cols_str) or any("년월" in x for x in cols_str)

    # ── 경우 1: 이미 헤더가 설정된 상태 (read_csv로 읽은 경우) ──
    if has_station_col and has_name_col and has_time_col:
        print("[INFO] 컬럼에 이미 지점/지점명/일시(년월) 있음 → 헤더 탐색 생략")
        print(f"[INFO] 헤더 컬럼: {cols_str}")
        return df_raw

    # ── 경우 2: header=None 으로 읽어서, 헤더 행을 위에서 찾아야 하는 경우 ──
    header_idx = None
    max_scan = min(20, len(df_raw))

    for i in range(max_scan):
        row_vals = [str(v).strip() for v in df_raw.iloc[i].tolist()]
        has_station = any(("지점" == x) or ("지점번호" in x) for x in row_vals)
        has_name = any("지점명" == x for x in row_vals)
        has_time = any(("일시" == x) or ("년월" in x) for x in row_vals)

        if has_station and has_name and has_time:
            header_idx = i
            break

    if header_idx is None:
        raise ValueError("헤더 행(지점/지점명/일시/년월)이 있는 줄을 찾지 못했음")

    cols = [str(v).strip() for v in df_raw.iloc[header_idx].tolist()]
    df = df_raw.iloc[header_idx + 1 :].copy()
    df.columns = cols

    # 다시 한 번 완전 빈 행 제거
    df = df.dropna(how="all")
    if df.empty:
        raise ValueError("헤더 설정 후 데이터가 비어 있음")

    print(f"[INFO] 헤더 행 index: {header_idx}")
    print(f"[INFO] 헤더 컬럼: {list(df.columns)}")
    return df


# ─────────────────────────────────────
# 4. 컬럼명 정리 (부분 문자열 기반 매핑)
#    - 유주님 ASOS 월자료 컬럼 중 "괜찮은 것들" 전부 매핑
# ─────────────────────────────────────


def map_kma_col_to_analytic(col: str) -> Optional[str]:
    """
    기상청 월자료 컬럼명을 분석용 컬럼명으로 매핑한다.
    - 완전 일치가 아니라, 부분 문자열 기준으로 매핑한다.
    - 여기서 매핑되는 컬럼들은 전부 시·도 평균 집계에 사용된다.
    """
    s = str(col).strip()

    # '나타난날'이 들어간 날짜 컬럼은 모두 스킵 (yyyymmdd)
    if "나타난날" in s:
        return None

    # ── id/이름/시간 ─────────────────
    if s == "지점" or "지점번호" in s:
        return "station_id"
    if s == "지점명":
        return "station_name"
    if s == "일시" or "년월" in s:
        return "ym"

    # ── 기온 관련 ───────────────────
    # 평균기온(°C)
    if "평균기온" in s and "최고" not in s and "최저" not in s:
        return "t_mean_month"
    # 평균최고기온(°C)
    if "평균최고기온" in s:
        return "t_max_mean_month"
    # 평균최저기온(°C)
    if "평균최저기온" in s:
        return "t_min_mean_month"
    # 최고기온(°C)  → (°C)가 붙은 실제 온도 컬럼만
    if "최고기온(" in s and "°C" in s:
        return "t_abs_max_month"
    # 최저기온(°C)
    if "최저기온(" in s and "°C" in s:
        return "t_abs_min_month"

    # ── 기압 관련 ───────────────────
    if "평균현지기압" in s:
        return "stn_pressure_mean_month"
    if "평균해면기압" in s:
        return "mslp_mean_month"
    if "최고해면기압" in s:
        return "mslp_max_month"
    if "최저해면기압" in s:
        return "mslp_min_month"

    # ── 수증기압/이슬점/습도 ────────
    if "평균수증기압" in s:
        return "vapor_pressure_mean_month"
    if "평균이슬점온도" in s:
        return "td_mean_month"
    if "평균상대습도" in s:
        return "rh_mean_month"
    # 데이터 컬럼명: "최소상대습도(%)"
    if "최소상대습도" in s:
        return "rh_min_month"

    # ── 강수 관련 ───────────────────
    # 월합강 수량(00~24h만)(mm)  ← 공백 들어가도 잡히도록
    if "월합강" in s and "수량" in s:
        return "precip_sum_month"
    # 일최다강수량(mm)
    if "일최다강수량(" in s:
        return "precip_max_day_month"
    # 1시간최다강수량(mm)
    if "1시간최다강수량(" in s:
        return "precip_max_1h_month"
    # 10분최다강수량(mm)
    if "10분최다강수량(" in s:
        return "precip_max_10min_month"

    # ── 풍속 관련 ───────────────────
    if "평균풍속" in s:
        return "wind_mean_month"
    if "최대풍속(" in s and "순간" not in s:
        return "wind_max_month"
    if "최대순간풍속(" in s:
        return "wind_gust_max_month"

    # ── 운량/일조/일사 ──────────────
    if "평균운량" in s and "중하층" not in s:
        return "cloud_mean_month"
    if "평균중하층운량" in s:
        return "cloud_midlow_mean_month"
    if "합계 일조시간" in s:
        return "sunshine_duration_sum_month"
    if "일조율" in s:
        return "sunshine_rate_month"
    if "합계 일사량" in s:
        return "solar_radiation_sum_month"

    # ── 적설 관련 ───────────────────
    if "최심적설(" in s and "신적설" not in s:
        return "snow_depth_max_month"
    if "최심신적설(" in s:
        return "snow_new_depth_max_month"
    if "3시간신적설합" in s:
        return "snow_new_3h_sum_month"

    # ── 초상/지면/지중온도 ───────────
    if "평균 최저초상온도" in s:
        return "frost_temp_min_mean_month"
    if s == "최저초상온도(°C)":
        return "frost_temp_min_month"
    if "평균지면온도" in s:
        return "ground_temp_mean_month"
    if "0.05m평균지중온도" in s:
        return "soil_temp_005m_mean_month"
    if "0.1m평균지중온도" in s:
        return "soil_temp_01m_mean_month"
    if "0.2m평균지중온도" in s:
        return "soil_temp_02m_mean_month"
    if "0.3m평균지중온도" in s:
        return "soil_temp_03m_mean_month"
    if "0.5m평균지중온도" in s:
        return "soil_temp_05m_mean_month"
    if "1.0m평균지중온도" in s:
        return "soil_temp_10m_mean_month"
    if "1.5m평균지중온도" in s:
        return "soil_temp_15m_mean_month"
    if "3.0m평균지중온도" in s:
        return "soil_temp_30m_mean_month"
    if "5.0m평균지중온도" in s:
        return "soil_temp_50m_mean_month"

    # 필요하면 여기 아래에 더 매핑 추가 가능
    return None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    원본 헤더를 분석용 컬럼명으로 rename 하고,
    사용할 수 있는 컬럼들만 필터링한다.
    - station_id / station_name / ym + 인식된 기상 변수 컬럼만 남긴다.
    - rename 과정에서 같은 분석용 이름이 여러 번 생기면 첫 번째 것만 사용한다.
    """
    analytic_cols: dict[str, str] = {}

    for col in df.columns:
        mapped = map_kma_col_to_analytic(col)
        if mapped is not None:
            analytic_cols[col] = mapped

    if "station_name" not in analytic_cols.values():
        print("[경고] station_name(지점명) 매핑 실패 가능성 있음", file=sys.stderr)

    if "ym" not in analytic_cols.values():
        print("[경고] ym(년월/일시) 매핑 실패 가능성 있음", file=sys.stderr)

    # rename 후 동일 이름 컬럼이 여러 개 생길 수 있음 → 첫 번째만 유지
    df = df.rename(columns=analytic_cols)
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # station_id / station_name / ym + 인식된 기상 변수만 유지
    base_cols = ["station_id", "station_name", "ym"]
    value_cols = [c for c in df.columns if c not in base_cols]

    keep_cols = [c for c in base_cols + value_cols if c in df.columns]
    df = df[keep_cols].copy()

    print(f"[INFO] 정리 후 컬럼: {list(df.columns)}")
    return df


# ─────────────────────────────────────
# 5. 지점명 -> 시도명 매핑 + 날짜 파싱
# ─────────────────────────────────────


def add_sido_and_time(df: pd.DataFrame) -> pd.DataFrame:
    if "station_name" not in df.columns:
        raise ValueError("station_name 컬럼이 없음 (지점명 매핑 확인 필요)")

    # 시도 매핑
    df["station_name"] = df["station_name"].astype(str).str.strip()
    df["sido_name"] = df["station_name"].map(station_to_sido)
    df["sido_name"] = df["sido_name"].fillna("기타")

    if "ym" not in df.columns:
        raise ValueError("ym 컬럼이 없음 (일시/년월 매핑 확인 필요)")

    ym_str = df["ym"].astype(str).str.strip()

    # 201801 같이 6자리 숫자인 경우 2018-01로 변환
    ym_str = ym_str.apply(
        lambda x: f"{x[:4]}-{x[4:6]}" if (len(x) == 6 and x.isdigit()) else x
    )

    ym_dt = pd.to_datetime(ym_str, errors="coerce")
    if ym_dt.isna().all():
        raise ValueError("ym 컬럼을 datetime으로 변환하지 못했음 (포맷 확인 필요)")

    df["year"] = ym_dt.dt.year
    df["month"] = ym_dt.dt.month
    df["ym"] = ym_dt.dt.strftime("%Y-%m")

    # 기간 필터 (2018-01 ~ 2024-12)
    mask = (df["ym"] >= "2018-01") & (df["ym"] <= "2024-12")
    df = df.loc[mask].copy()

    return df


# ─────────────────────────────────────
# 6. 시·도 단위 집계
#    - 인식된 모든 숫자형 기상 변수 컬럼을 평균으로 집계
# ─────────────────────────────────────


def aggregate_to_sido(df: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["station_id", "station_name", "sido_name", "ym", "year", "month"]

    # 숫자형으로 강제 변환
    for c in df.columns:
        if c not in key_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 집계 대상: key_cols 제외한 나머지 숫자형 컬럼 전부
    value_cols = [
        c
        for c in df.columns
        if c not in key_cols and pd.api.types.is_numeric_dtype(df[c])
    ]

    if not value_cols:
        raise ValueError("집계할 숫자형 기상 변수 컬럼이 없음")

    agg_dict = {c: "mean" for c in value_cols}

    grouped = df.groupby(["sido_name", "ym"], as_index=False).agg(agg_dict)

    grouped["year"] = grouped["ym"].str.slice(0, 4).astype(int)
    grouped["month"] = grouped["ym"].str.slice(5, 7).astype(int)

    ordered_cols = ["year", "month", "ym", "sido_name"] + value_cols
    grouped = grouped[ordered_cols].sort_values(["sido_name", "year", "month"])

    return grouped


# ─────────────────────────────────────
# 7. 메인 실행부
# ─────────────────────────────────────


def main():
    # 1) ASOS 원본 로드 + 헤더 행 결정
    df_raw = load_raw_asos_excel(raw_path)
    df_headed = find_header_and_build(df_raw)

    # 2) 컬럼명 정리 (매핑 가능한 컬럼 모두 사용)
    df_norm = normalize_columns(df_headed)

    # 3) 시도 + 시간 정보 추가
    df_sido = add_sido_and_time(df_norm)

    # 4) 시·도 월평균 집계
    df_agg = aggregate_to_sido(df_sido)

    out_file = OUT_DIR / "asos_2018_2024_monthly.csv"
    df_agg.to_csv(out_file, index=False, encoding="utf-8-sig")

    print(f"[완료] 저장: {out_file}")
    print(df_agg.head())


if __name__ == "__main__":
    main()
