"""
AirKorea PM2.5(초미세먼지) 일평균 엑셀 자동 다운로드 스크립트 (시도별, 월별 저장)

- 에어코리아 미세먼지 세부 측정정보(pmRelay)의 "엑셀" 버튼이 보내는 요청을
  requests로 그대로 흉내 내서 엑셀 파일을 내려받는다.
- 기간: AIRKOREA_PM25_START_YM ~ AIRKOREA_PM25_END_YM (YYYYMM, 기본: 2018-01 ~ 2024-12)
- 항목: itemCode = AIRKOREA_PM25_ITEM_CODE (기본 11008, PM2.5)
- 시도:
    서울, 강원, 경남, 부산, 울산, 경북, 대구,
    전남, 광주, 전북, 충남, 충북, 대전, 세종, 제주

- 각 시도에 대응하는 district 코드는 REGION_DISTRICT_MAP에 하드코딩되어 있으며,
  다운로드된 파일명에는 district가 전혀 노출되지 않는다.

- 다운로드 경로:
    data/raw/climate/airkorea/pm25
    예: data/raw/climate/airkorea/pm25/airkorea_pm25_서울_202401.xls
"""

import os
import time
import requests
from pathlib import Path

from dotenv import load_dotenv

# ─────────────────────────────────────────
# 0. 환경 변수 로드
# ─────────────────────────────────────────
load_dotenv()

# ─────────────────────────────────────────
# 1. 경로 설정
#    - ROOT_DIR: 레포 루트
#    - RAW_DIR : data/climate/raw/airkorea/pm25
# ─────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[3]

RAW_DIR = ROOT_DIR / "data" / "raw" / "climate" / "airkorea" / "pm25"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────
# 2. 다운로드 URL 및 HTTP 헤더 (PM2.5 전용)
# ─────────────────────────────────────────
EXCEL_URL = os.getenv(
    "AIRKOREA_PM25_EXCEL_URL",
    "https://www.airkorea.or.kr/web/pmrelayExcel",
)

headers = {
    "Accept": os.getenv(
        "AIRKOREA_PM25_ACCEPT",
        "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,"
        "image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    ),
    "Accept-Encoding": os.getenv(
        "AIRKOREA_PM25_ACCEPT_ENCODING", "gzip, deflate, br, zstd"
    ),
    "Accept-Language": os.getenv(
        "AIRKOREA_PM25_ACCEPT_LANGUAGE",
        "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    ),
    "Cache-Control": os.getenv("AIRKOREA_PM25_CACHE_CONTROL", "max-age=0"),
    "Connection": os.getenv("AIRKOREA_PM25_CONNECTION", "keep-alive"),
    "Content-Type": os.getenv(
        "AIRKOREA_PM25_CONTENT_TYPE",
        "application/x-www-form-urlencoded",
    ),
    "Origin": os.getenv("AIRKOREA_PM25_ORIGIN", "https://www.airkorea.or.kr"),
    "Referer": os.getenv(
        "AIRKOREA_PM25_REFERER",
        "https://www.airkorea.or.kr/web/pmRelay?itemCode=11008&pMENU_NO=109",
    ),
    "Upgrade-Insecure-Requests": os.getenv(
        "AIRKOREA_PM25_UPGRADE_INSECURE_REQUESTS", "1"
    ),
    "User-Agent": os.getenv(
        "AIRKOREA_PM25_USER_AGENT",
        (
            "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/142.0.0.0 Mobile Safari/537.36"
        ),
    ),
    "sec-ch-ua": os.getenv(
        "AIRKOREA_PM25_SEC_CH_UA",
        '"Chromium";v="142", "Google Chrome";v="142", "Not_A Brand";v="99"',
    ),
    "sec-ch-ua-mobile": os.getenv("AIRKOREA_PM25_SEC_CH_UA_MOBILE", "?1"),
    "sec-ch-ua-platform": os.getenv("AIRKOREA_PM25_SEC_CH_UA_PLATFORM", '"Android"'),
    "Sec-Fetch-Dest": os.getenv("AIRKOREA_PM25_SEC_FETCH_DEST", "document"),
    "Sec-Fetch-Mode": os.getenv("AIRKOREA_PM25_SEC_FETCH_MODE", "navigate"),
    "Sec-Fetch-Site": os.getenv("AIRKOREA_PM25_SEC_FETCH_SITE", "same-origin"),
    "Sec-Fetch-User": os.getenv("AIRKOREA_PM25_SEC_FETCH_USER", "?1"),
    "Cookie": os.getenv("AIRKOREA_PM25_COOKIE", ""),
}

# ─────────────────────────────────────────
# 3. 기간 설정 (월 루프, PM2.5 전용)
# ─────────────────────────────────────────
START_YM = os.getenv("AIRKOREA_PM25_START_YM", "201801")
END_YM = os.getenv("AIRKOREA_PM25_END_YM", "202412")

if len(START_YM) != 6 or len(END_YM) != 6:
    raise ValueError(
        "AIRKOREA_PM25_START_YM / AIRKOREA_PM25_END_YM 은 YYYYMM 형식이어야 한다."
    )

start_year = int(START_YM[:4])
start_month = int(START_YM[4:6])
end_year = int(END_YM[:4])
end_month = int(END_YM[4:6])


def month_iter(y0: int, m0: int, y1: int, m1: int):
    """
    시작 (y0, m0) ~ 끝 (y1, m1) 구간을 월 단위로 순회하는 제너레이터.
    각 단계에서 (year, month, yyyymm_str)을 반환한다.
    """
    y = y0
    m = m0
    while (y < y1) or (y == y1 and m <= m1):
        yyyymm = f"{y}{m:02d}"
        yield y, m, yyyymm
        m += 1
        if m > 12:
            m = 1
            y += 1


# ─────────────────────────────────────────
# 4. itemCode / 메뉴 번호 (PM2.5 전용)
# ─────────────────────────────────────────
ITEM_CODE = os.getenv("AIRKOREA_PM25_ITEM_CODE", "11008")  # PM2.5
PMENU_NO = os.getenv("AIRKOREA_PM25_PMENU_NO", "109")

# ─────────────────────────────────────────
# 5. 시도별 district 코드 매핑 (하드코딩)
# ─────────────────────────────────────────
REGION_DISTRICT_MAP = {
    # "한글 시도명": "district코드"
    "서울": "02",
    "경기": "031",
    "인천": "032",
    "강원": "033",
    "충남": "041",
    "대전": "042",
    "충북": "043",
    "부산": "051",
    "울산": "052",
    "대구": "053",
    "경북": "054",
    "경남": "055",
    "전남": "061",
    "광주": "062",
    "전북": "063",
    "제주": "064",
    "세종": "044",
}

# ─────────────────────────────────────────
# 6. 자동 다운로드 루프
# ─────────────────────────────────────────
for region_kor, district in REGION_DISTRICT_MAP.items():
    for year, month, yyyymm in month_iter(start_year, start_month, end_year, end_month):
        yyyy_str = str(year)
        mm_str = f"{month:02d}"
        search_date = f"{yyyy_str}-{mm_str}-01"  # 해당 월 1일

        # 쿼리스트링
        query_params = {
            "strDateDiv": "2",  # 일평균
            "searchDate": search_date,
            "district": district,
            "itemCode": ITEM_CODE,
            "searchDate_f": yyyymm,
        }

        # 폼 데이터
        payload = {
            "strDateDiv": "2",
            "searchDate": search_date,
            "district": district,
            "itemCode": ITEM_CODE,
            "searchDate_f": yyyymm,
            "pMENU_NO": PMENU_NO,
            "yyyy": yyyy_str,
            "mm": mm_str,
            "searchDate_yyyy": yyyy_str,
            "searchDate_mm": mm_str,
        }

        print(
            f"[PM2.5 다운로드 시도] 시도={region_kor}, district={district}, yyyymm={yyyymm}"
        )

        try:
            resp = requests.post(
                EXCEL_URL,
                headers=headers,
                params=query_params,
                data=payload,
                timeout=30,
            )
        except Exception as e:
            print(f"  !! 요청 에러 발생: {e}")
            time.sleep(1)
            continue

        ctype = resp.headers.get("Content-Type", "")
        if resp.status_code != 200 or "application" not in ctype:
            print(
                f"  !! 실패: status={resp.status_code}, "
                f"Content-Type={ctype!r}, text={resp.text[:200]!r}",
            )
            time.sleep(1)
            continue

        # 파일명: data/raw/climate/airkorea/pm25/airkorea_pm25_서울_202401.xls
        fname = RAW_DIR / f"airkorea_pm25_{region_kor}_{yyyymm}.xls"
        try:
            with open(fname, "wb") as f:
                f.write(resp.content)
        except Exception as e:
            print(f"  !! 파일 저장 실패: {e}")
            time.sleep(1)
            continue

        print(f"  -> 저장 완료: {fname}")
        time.sleep(1.0)

print("PM2.5 다운로드 루프 종료")
