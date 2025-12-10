from __future__ import annotations
from typing import Literal
from fastapi import APIRouter, HTTPException, Query
from core.monthly_map_logic import (
    get_monthly_map_rows,
    get_available_year_months,
)


router = APIRouter(
    prefix="/api",
    tags=["monthly"],
)


@router.get("/monthly_map")
def monthly_map(
    ym: str = Query(..., description="조회 연월 (형식: YYYY-MM, 예: 2023-12)"),
    metric: Literal["actual", "pred", "error_pct"] = "actual",
):
    """
    월 단위 시도별 실제/예측/오차율 맵 데이터를 반환하는 엔드포인트.

    예)
      GET /api/monthly_map?ym=2023-12&metric=actual
    """
    # 1) ym -> year, month 파싱 및 검증
    try:
        parts = ym.split("-")
        if len(parts) != 2:
            raise ValueError

        year = int(parts[0])
        month = int(parts[1])

        if not (1 <= month <= 12):
            raise ValueError
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="ym 파라미터 형식이 잘못되었습니다. 'YYYY-MM' 형식으로 보내세요. 예: 2023-12",
        )

    # 2) 로직 호출
    try:
        rows = get_monthly_map_rows(year=year, month=month, metric=metric)
    except ValueError as e:
        # 로직에서 year/month/metric 문제로 ValueError 발생 시 400으로 변환
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"월 단위 맵 데이터를 생성하는 중 오류가 발생했습니다: {e}",
        )

    # 3) 응답
    return {
        "ym": ym,
        "year": year,
        "month": month,
        "metric": metric,
        "data": rows,
    }


@router.get("/monthly_map/available")
def monthly_available():
    """
    사용 가능한 (year, month) 리스트를 반환하는 엔드포인트.

    예)
      GET /api/monthly_map/available

    응답 예:
      {
        "items": [
          {"year": 2018, "month": 1},
          {"year": 2018, "month": 2},
          ...
        ]
      }
    """
    items = get_available_year_months()
    return {"items": items}
