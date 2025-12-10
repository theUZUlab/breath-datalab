import os
from pathlib import Path
from typing import List, Optional
from urllib.parse import unquote
import requests
from requests import HTTPError
import pandas as pd
from dotenv import load_dotenv

# ─────────────────────────────────────
# 0. 프로젝트 루트 및 .env 로드
# ─────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[2]  # .../breath-datalab
ENV_PATH = BASE_DIR / ".env"

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

# ─────────────────────────────────────
# 1. 공통 유틸
# ─────────────────────────────────────
KOREAN_SIDO_LIST: List[str] = [
    "서울",
    "부산",
    "대구",
    "인천",
    "광주",
    "대전",
    "울산",
    "경기",
    "강원",
    "충북",
    "충남",
    "전북",
    "전남",
    "경북",
    "경남",
    "제주",
    "세종",
]


def _get_env_or_raise(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"환경 변수 {name} 가 비어 있습니다. .env 에 {name}=... 을 추가하세요."
        )
    return value


# ─────────────────────────────────────
# 2. 에어코리아 – 시도별 실시간 측정정보
#    get_daily_risk 에 들어갈 env_obs_df 형태: ['sido_name', 'pm25', 'pm10']
# ─────────────────────────────────────
def fetch_airkorea_by_sido(sido_name: str) -> pd.DataFrame:
    """
    특정 시도(sido_name)에 대해
    '시도별 실시간 측정정보 조회(getCtprvnRltmMesureDnsty)'를 호출해서
    측정소별 row가 들어 있는 DataFrame을 반환한다.
    """
    service_key_raw = _get_env_or_raise("AIRKOREA_SERVICE_KEY")
    # 인코딩 키든 디코딩 키든 safe
    service_key = unquote(service_key_raw, "utf-8")

    url = "http://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getCtprvnRltmMesureDnsty"

    params = {
        "serviceKey": service_key,
        "returnType": "json",
        "numOfRows": "100",
        "pageNo": "1",
        "sidoName": sido_name,
        "ver": "1.0",
    }

    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    items = data["response"]["body"]["items"]
    df = pd.DataFrame(items)

    # 숫자형 컬럼 변환
    for col in ["pm10Value", "pm25Value"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def build_airkorea_env_obs_for_all_sido() -> Optional[pd.DataFrame]:
    """
    전 시도에 대해 에어코리아 실시간 값을 조회해서
    get_daily_risk 의 env_obs_df 로 바로 쓸 수 있는 형태로 만든다.

    반환 컬럼:
    - sido_name
    - pm25
    - pm10

    모든 시도에서 데이터를 가져오지 못한 경우(None)를 반환하여
    상위 로직이 '환경데이터 없이' 계산하도록 한다.
    """
    rows = []

    for sido in KOREAN_SIDO_LIST:
        try:
            df = fetch_airkorea_by_sido(sido)
        except HTTPError as e:
            # 레이트 리밋(429) 등 HTTP 에러 처리
            if e.response is not None and e.response.status_code == 429:
                print(
                    f"[WARN] AirKorea API 429 Too Many Requests (sido={sido}), skip this region"
                )
                continue
            # 그 외 HTTP 에러는 그대로 올린다
            raise

        if df.empty:
            # 이 시도에 데이터가 없으면 스킵
            continue

        row = {
            "sido_name": sido,
            "pm10": df["pm10Value"].mean(skipna=True),
            "pm25": df["pm25Value"].mean(skipna=True),
        }
        rows.append(row)

    if not rows:
        # 전 시도 모두 실패/빈 경우 → 상위에서 fallback 처리할 수 있도록 None 반환
        print(
            "[WARN] AirKorea API 결과가 모두 비어 있습니다. env_obs_df 없이 위험지수 계산을 진행합니다."
        )
        return None

    env_df = pd.DataFrame(rows)
    return env_df


# ─────────────────────────────────────
# 3. (옵션) 기상청 ASOS 일자료 – 예시
# ─────────────────────────────────────
def fetch_kma_daily_asos_raw(tm: str, stn: int = 0) -> pd.DataFrame:
    """
    기상청 지상(ASOS) 일자료 예시 호출 함수.
    tm: 'YYYYMMDD'
    stn: 지점번호 (0 이면 전체)
    """
    from io import StringIO

    auth_key = _get_env_or_raise("KMA_ASOS_AUTH_KEY")

    url = "https://apihub.kma.go.kr/api/typ01/url/kma_sfcdd.php"

    params = {
        "tm": tm,
        "stn": stn,
        "disp": 0,
        "help": 0,
        "authKey": auth_key,
    }

    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()

    text = resp.text
    df = pd.read_csv(StringIO(text))
    return df
