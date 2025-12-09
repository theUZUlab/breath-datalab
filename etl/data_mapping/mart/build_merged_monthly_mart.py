import pandas as pd
from pathlib import Path

# ------------------------------------
# 0. 파일 경로 설정
# ------------------------------------
BASE_DIR = Path("data")

HEALTH_PATH = BASE_DIR / "processed" / "health" / "health_J00_J99_monthly.csv"
AIR_PATH = BASE_DIR / "processed" / "climate" / "airkorea_pm25_pm10_monthly.csv"
ASOS_PATH = BASE_DIR / "processed" / "climate" / "asos_2018_2024_monthly.csv"
POP_PATH = BASE_DIR / "raw" / "population" / "kosis" / "population_monthly.xlsx"

OUTPUT_PATH = BASE_DIR / "mart" / "merged_monthly.csv"

# ------------------------------------
# 1. 원천 데이터 로드
# ------------------------------------
print("데이터 로드 중...")

health = pd.read_csv(HEALTH_PATH)
air = pd.read_csv(AIR_PATH)
asos = pd.read_csv(ASOS_PATH)
pop_raw = pd.read_excel(POP_PATH)

print(f"health: {health.shape}")
print(f"air   : {air.shape}")
print(f"asos  : {asos.shape}")
print(f"pop   : {pop_raw.shape}")

# ------------------------------------
# 2. 인구 데이터 정리 (가로 → 세로 변환)
#    - 통계청 양식: '행정구역(시군구)별' + '2018.01', '2018.01.1', '2018.01.2' ...
#    - 여기서는 '총인구수 (명)'에 해당하는 컬럼만 사용 ('.1', '.2'는 남/여라서 제외)
# ------------------------------------
print("인구 데이터 전처리...")

# 2-1) 첫 번째 행(헤더 설명 행) 제거
pop = pop_raw[pop_raw["행정구역(시군구)별"] != "행정구역(시군구)별"].copy()

# 2-2) 총인구수 컬럼만 선택 (뒤에 .1, .2 붙은 것은 남/여 인구라 제외)
keep_cols = ["행정구역(시군구)별"] + [
    c
    for c in pop.columns
    if c != "행정구역(시군구)별" and not c.endswith(".1") and not c.endswith(".2")
]

pop_sel = pop[keep_cols].copy()

# 2-3) wide → long (행정구역, ym_orig, population)
pop_long = pop_sel.melt(
    id_vars="행정구역(시군구)별", var_name="ym_orig", value_name="population"
)

# 2-4) ym_orig = '2018.01' → year, month, ym 변환
pop_long["year"] = pop_long["ym_orig"].str.split(".").str[0].astype(int)
pop_long["month"] = pop_long["ym_orig"].str.split(".").str[1].astype(int)
pop_long["ym"] = (
    pop_long["year"].astype(str) + "-" + pop_long["month"].astype(str).str.zfill(2)
)

# 2-5) 시도명 매핑
region_map = {
    "서울특별시": "서울",
    "부산광역시": "부산",
    "대구광역시": "대구",
    "인천광역시": "인천",
    "광주광역시": "광주",
    "대전광역시": "대전",
    "울산광역시": "울산",
    "세종특별자치시": "세종",
    "경기도": "경기",
    "강원특별자치도": "강원",
    "충청북도": "충북",
    "충청남도": "충남",
    "전라남도": "전남",
    "전북특별자치도": "전북",
    "경상북도": "경북",
    "경상남도": "경남",
    "제주특별자치도": "제주",
}

pop_long["sido_raw"] = pop_long["행정구역(시군구)별"]
pop_long["sido_name"] = pop_long["sido_raw"].map(region_map)

# 전국 / 매핑 안 되는 값 제거
pop_long = pop_long[pop_long["sido_name"].notna()].copy()

# 2-6) population 숫자형 변환
pop_long["population"] = pd.to_numeric(pop_long["population"], errors="coerce")

print("인구 데이터 정리 결과:")
print(pop_long[["year", "month", "ym", "sido_name", "population"]].head())

# ------------------------------------
# 3. 기준 연도 범위 설정 (health 연도 기준으로 자르기)
# ------------------------------------
min_year = int(health["year"].min())
max_year = int(health["year"].max())

air_sub = air[(air["year"] >= min_year) & (air["year"] <= max_year)].copy()
asos_sub = asos[(asos["year"] >= min_year) & (asos["year"] <= max_year)].copy()
pop_sub = pop_long[
    (pop_long["year"] >= min_year) & (pop_long["year"] <= max_year)
].copy()

# ------------------------------------
# 4. 컬럼 슬림하게 선택 (필요한 컬럼만)
# ------------------------------------
health_sel = health[["year", "month", "ym", "sido_name", "resp_total"]].copy()

air_sel = air_sub[
    [
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
].copy()

asos_sel = asos_sub[
    [
        "year",
        "month",
        "ym",
        "sido_name",
        "t_mean_month",
        "rh_mean_month",
        "precip_sum_month",
        "wind_mean_month",
    ]
].copy()

pop_sel2 = pop_sub[["year", "month", "ym", "sido_name", "population"]].copy()

# ------------------------------------
# 5. health를 기준으로 LEFT JOIN으로 통합
#    - health에 있는 모든 (year, month, sido) 행은 유지
#    - air/asos/pop에 없으면 NaN으로 남김
# ------------------------------------
key_cols = ["year", "month", "ym", "sido_name"]

print("health + air 머지...")
df = health_sel.merge(air_sel, on=key_cols, how="left")

print("+ asos 머지...")
df = df.merge(asos_sel, on=key_cols, how="left")

print("+ population 머지...")
df = df.merge(pop_sel2, on=key_cols, how="left")

print("통합 결과 shape:", df.shape)

# ------------------------------------
# 6. 인구 보정: 10만 명당 환자수(resp_rate_total) 계산
#    resp_rate_total = resp_total / population * 100000
# ------------------------------------
df["resp_rate_total"] = df["resp_total"] / df["population"] * 100000

# ------------------------------------
# 7. 정렬 및 저장
# ------------------------------------
df = df.sort_values(["sido_name", "year", "month"]).reset_index(drop=True)

df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

print("저장 완료:", OUTPUT_PATH.resolve())
print("컬럼 목록:")
print(df.columns.tolist())
