from __future__ import annotations
from fastapi import APIRouter, HTTPException, Query
from backend.core.daily_risk_logic import build_daily_risk_rows_for_api
from backend.core.env_client import validate_date_format

router = APIRouter(
    prefix="/api",
    tags=["daily"],
)


@router.get("/daily_risk")
def daily_risk(
    date: str = Query(..., description="조회할 날짜 (형식: YYYY-MM-DD)"),
):
    """
    일 단위 시도별 위험점수/등급 맵 데이터를 반환하는 엔드포인트.

    예)
      GET /api/daily_risk?date=2024-01-01

    응답 형식 예:
      {
        "date": "2024-01-01",
        "env_used": true,
        "data": [
          {
            "sido_name": "서울",
            "value": 78.5,
            "extra": {
              "risk_score": 78.5,
              "risk_level": "High",
              "date": "2024-01-01"
            }
          },
          ...
        ]
      }
    """
    # 1) 날짜 형식 검증
    try:
        validate_date_format(date)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 2) 위험지수 계산
    #    env_obs_df 를 None 으로 넘기면, 내부에서 AirKorea 실시간 API를 호출해 사용함.
    try:
        rows = build_daily_risk_rows_for_api(date_str=date, env_obs_df=None)
    except ValueError as e:
        # 예: 해당 날짜에 대응되는 월 예측값이 없을 때 등
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"일 단위 위험지수를 계산하는 중 오류가 발생했습니다: {e}",
        )

    # 3) env_used 플래그
    #    현재 구현 기준으로는 AirKorea API를 사용하므로, rows 가 비어있지 않으면 True 로 본다.
    env_used = bool(rows)

    return {
        "date": date,
        "env_used": env_used,
        "data": rows,
    }
