"""
HIRA 4단질병 요양기관 소재지별 엑셀 자동 다운로드 스크립트

- HIRA_OPEN API 페이지의 엑셀 다운로드 폼을 그대로 requests로 호출해 엑셀을 내려받는다.
- 진료년월(2018-01 ~ 2024-12) 범위 전체에 대해 J00~J99 상병코드를 루프한다.
- 다운로드 경로:
    프로젝트 루트 기준 data/hira/{HIRA_OUTPUT_DIR}/J00_201801_202012.xlsx 형태로 저장된다.
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
#    - ROOT_DIR: repo 최상위 디렉터리 (etl/ 기준으로 두 단계 위)
#    - RAW_DIR : data/hira/{HIRA_OUTPUT_DIR} (엑셀 저장 위치)
# ─────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[3]

HIRA_OUTPUT_DIR_NAME = os.getenv("HIRA_OUTPUT_DIR", "hira_4th_disease")
RAW_DIR = ROOT_DIR / "data" / "raw" / "health" / "hira" / HIRA_OUTPUT_DIR_NAME
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────
# 2. 다운로드 URL 및 HTTP 헤더
#    - 민감/환경 의존 값은 .env에서 불러옴
# ─────────────────────────────────────────
EXCEL_URL = os.getenv(
    "HIRA_EXCEL_URL",
    "https://opendata.hira.or.kr/op/opc/downExcel4thDsInfo.do",
)

headers = {
    "Accept": os.getenv(
        "HIRA_ACCEPT",
        "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,"
        "image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    ),
    "Accept-Encoding": os.getenv("HIRA_ACCEPT_ENCODING", "gzip, deflate, br, zstd"),
    "Accept-Language": os.getenv(
        "HIRA_ACCEPT_LANGUAGE", "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7"
    ),
    "Cache-Control": os.getenv("HIRA_CACHE_CONTROL", "max-age=0"),
    "Connection": os.getenv("HIRA_CONNECTION", "keep-alive"),
    "Content-Type": os.getenv("HIRA_CONTENT_TYPE", "application/x-www-form-urlencoded"),
    "Origin": os.getenv("HIRA_ORIGIN", "https://opendata.hira.or.kr"),
    "Referer": os.getenv(
        "HIRA_REFERER", "https://opendata.hira.or.kr/op/opc/olap4thDsInfoTab5.do"
    ),
    "Upgrade-Insecure-Requests": os.getenv("HIRA_UPGRADE_INSECURE_REQUESTS", "1"),
    "User-Agent": os.getenv(
        "HIRA_USER_AGENT",
        (
            "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/142.0.0.0 Mobile Safari/537.36"
        ),
    ),
    "sec-ch-ua": os.getenv(
        "HIRA_SEC_CH_UA",
        '"Chromium";v="142", "Google Chrome";v="142", "Not_A Brand";v="99"',
    ),
    "sec-ch-ua-mobile": os.getenv("HIRA_SEC_CH_UA_MOBILE", "?1"),
    "sec-ch-ua-platform": os.getenv("HIRA_SEC_CH_UA_PLATFORM", '"Android"'),
    "Sec-Fetch-Dest": os.getenv("HIRA_SEC_FETCH_DEST", "document"),
    "Sec-Fetch-Mode": os.getenv("HIRA_SEC_FETCH_MODE", "navigate"),
    "Sec-Fetch-Site": os.getenv("HIRA_SEC_FETCH_SITE", "same-origin"),
    "Sec-Fetch-User": os.getenv("HIRA_SEC_FETCH_USER", "?1"),
    "Cookie": os.getenv("HIRA_COOKIE", ""),
}

# ─────────────────────────────────────────
# 3. 상병코드 / 기간 설정
#    - 상병코드: J00 ~ J99
#    - 기간 블록: 2018-01~2020-12, 2021-01~2023-12, 2024-01~2024-12
# ─────────────────────────────────────────
# J00 ~ J99
codes = [f"J{str(i).zfill(2)}" for i in range(0, 100)]

# 진료년월(YYYYMM) 기준 기간 블록
periods = [
    ("201801", "202012"),  # 2018/01 ~ 2020/12
    ("202101", "202312"),  # 2021/01 ~ 2023/12
    ("202401", "202412"),  # 2024/01 ~ 2024/12
]

# ─────────────────────────────────────────
# 4. 기본 Form Data 템플릿
#    - 실제 요청 시 코드/기간에 맞게 값 덮어씌운다.
# ─────────────────────────────────────────
base_payload = {
    "searchWrd": "급성 비인두염[감기]",  # 루프에서 코드별로 덮어씀
    "olapCd": "AJ00",  # 루프에서 'A' + J코드로 덮어씀
    "olapCdNm": "급성 비인두염[감기]",  # 루프에서 코드 이름으로 덮어씀
    "tabGubun": "Tab5",
    "gubun": "D",
    "sRvYr": "2018",  # 루프에서 시작 연도로 덮어씀
    "eRvYr": "2020",  # 루프에서 끝 연도로 덮어씀
    "sDiagYm": "201801",  # 루프에서 시작 진료년월로 덮어씀
    "eDiagYm": "202012",  # 루프에서 끝 진료년월로 덮어씀
    "sYm": "2018-01",  # 루프에서 YYYY-MM로 변환하여 덮어씀
    "eYm": "2020-12",
}

# ─────────────────────────────────────────
# 5. 자동 다운로드 루프
#    - 코드 × 기간 블록 조합마다 POST 요청 후 엑셀 저장
# ─────────────────────────────────────────
for code in codes:
    for start, end in periods:
        s_year = start[:4]
        e_year = end[:4]

        payload = base_payload.copy()

        # 상병코드: AJ00, AJ01, ... AJ99
        payload["olapCd"] = "A" + code

        # 검색어/이름: 일단 코드명으로 설정
        payload["searchWrd"] = code
        payload["olapCdNm"] = code

        # 기간 관련 값 덮어쓰기
        payload["sRvYr"] = s_year
        payload["eRvYr"] = e_year
        payload["sDiagYm"] = start
        payload["eDiagYm"] = end
        payload["sYm"] = f"{s_year}-{start[4:6]}"
        payload["eYm"] = f"{e_year}-{end[4:6]}"

        print(f"[다운로드 시도] {code} {start}~{end}")

        resp = requests.post(EXCEL_URL, headers=headers, data=payload)
        ctype = resp.headers.get("Content-Type", "")

        if resp.status_code != 200 or "application" not in ctype:
            print(f"  !! 실패: status={resp.status_code}, Content-Type={ctype}")
            time.sleep(1)
            continue

        # 파일명: J00_201801_202012.xlsx 형식
        fname = RAW_DIR / f"{code}_{start}_{end}.xlsx"
        with open(fname, "wb") as f:
            f.write(resp.content)

        print(f"  -> 저장 완료: {fname}")
        time.sleep(1.0)

print("모든 다운로드 루프 종료")
