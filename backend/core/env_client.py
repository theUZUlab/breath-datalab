from __future__ import annotations
from typing import Optional
from datetime import datetime
import pandas as pd


"""
환경 관측값(대기질 + 기상)을 불러와서
get_daily_risk(...) 에 넘길 env_obs_df 를 만드는 모듈이다.

현재 버전:
- 실제 공개 API 연동은 아직 안 되어 있고,
- 구조/인터페이스만 맞춰 놓은 상태다.
- 나중에 에어코리아 / 기상청 API를 붙일 때 이 파일만 수정하면 된다.

env_obs_df 기대 스키마:
    columns: ['sido_name', 'pm25', 'pm10', 't', 'rh']

    예:
        sido_name,pm25,pm10,t,rh
        서울,35,55,2.3,45
        부산,22,40,5.0,50
"""


# ─────────────────────────────────────
# 1) 유효한 날짜 형식 검증 유틸
# ─────────────────────────────────────


def validate_date_format(date_str: str) -> None:
    """
    'YYYY-MM-DD' 형식의 문자열인지 검증한다.
    형식이 잘못되면 ValueError를 발생시킨다.
    """
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise ValueError(
            f"잘못된 날짜 형식입니다. 'YYYY-MM-DD' 형식이어야 합니다: {date_str}"
        )


# ─────────────────────────────────────
# 2) TODO: 실제 에어코리아 / 기상청 API 연동 자리
# ─────────────────────────────────────


def fetch_air_quality_from_api(date_str: str) -> pd.DataFrame:
    """
    [TODO] 에어코리아(또는 유사 공개 API)를 호출해서,
    특정 날짜(date_str)에 대한 시도별 PM2.5 / PM10을 가져오는 함수.

    현재는 실제 API 호출을 하지 않고, 빈 DataFrame을 반환한다.
    나중에 구현 시, 반환 스키마는 다음 형태를 맞추면 된다.

    columns 예시:
        - 'sido_name' : 시/도 이름 (서울, 부산, ...)
        - 'pm25'      : 해당 날짜 시도 평균 PM2.5
        - 'pm10'      : 해당 날짜 시도 평균 PM10
    """
    validate_date_format(date_str)

    # TODO: 실제 에어코리아 API 연동 후, 아래와 같은 형태로 DataFrame 생성
    # data = [
    #     {"sido_name": "서울", "pm25": 35.0, "pm10": 55.0},
    #     {"sido_name": "부산", "pm25": 22.0, "pm10": 40.0},
    #     ...
    # ]
    # return pd.DataFrame(data)

    # 현재는 아직 API 연동 전이므로, 빈 DataFrame 리턴
    return pd.DataFrame(columns=["sido_name", "pm25", "pm10"])


def fetch_weather_from_api(date_str: str) -> pd.DataFrame:
    """
    [TODO] 기상청(또는 유사 공개 API)을 호출해서,
    특정 날짜(date_str)에 대한 시도별 기온/습도(t, rh)를 가져오는 함수.

    현재는 실제 API 호출을 하지 않고, 빈 DataFrame을 반환한다.
    나중에 구현 시, 반환 스키마는 다음 형태를 맞추면 된다.

    columns 예시:
        - 'sido_name' : 시/도 이름
        - 't'         : 해당 날짜 시도 평균 기온(℃)
        - 'rh'        : 해당 날짜 시도 평균 상대습도(%)
    """
    validate_date_format(date_str)

    # TODO: 실제 기상청 API 연동 후, 아래와 같은 형태로 DataFrame 생성
    # data = [
    #     {"sido_name": "서울", "t": 2.3, "rh": 45.0},
    #     {"sido_name": "부산", "t": 5.0, "rh": 50.0},
    #     ...
    # ]
    # return pd.DataFrame(data)

    # 현재는 아직 API 연동 전이므로, 빈 DataFrame 리턴
    return pd.DataFrame(columns=["sido_name", "t", "rh"])


# ─────────────────────────────────────
# 3) env_obs_df 빌더
# ─────────────────────────────────────


def build_env_obs_df(date_str: str) -> Optional[pd.DataFrame]:
    """
    특정 날짜(date_str)에 대해,
    대기질 + 기상 데이터를 머지해서 env_obs_df(DataFrame)를 만드는 함수.

    get_daily_risk(...) 에 넘길 수 있는 스키마:
        columns: ['sido_name', 'pm25', 'pm10', 't', 'rh']

    현재 버전 동작:
        - API 연동이 아직 안 되어 있어,
          air_df, weather_df 모두 빈 DataFrame을 반환한다.
        - 따라서 머지 결과도 빈 DataFrame이므로,
          실제 계산에서는 env_obs_df=None 으로 처리하는 것이 자연스럽다.
    """
    validate_date_format(date_str)

    air_df = fetch_air_quality_from_api(date_str)
    weather_df = fetch_weather_from_api(date_str)

    if air_df.empty and weather_df.empty:
        # 아직 API 연동 전이라면 None 리턴 → 시즌 weight + 기본 로직만 사용
        return None

    # sido_name 기준 inner join
    df = pd.merge(air_df, weather_df, on="sido_name", how="inner")

    # 스키마를 기대 형태로 정리
    expected_cols = ["sido_name", "pm25", "pm10", "t", "rh"]
    for col in expected_cols:
        if col not in df.columns:
            # 없는 컬럼은 NaN으로 채워 넣는다.
            df[col] = pd.NA

    df = df[expected_cols].copy()

    # 만약 join 결과가 비어 있으면 None을 돌려서
    # 호출 측에서 env_obs_df를 사용하지 않도록 할 수 있다.
    if df.empty:
        return None

    return df
