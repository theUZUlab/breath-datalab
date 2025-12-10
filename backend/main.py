from __future__ import annotations
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from core.config import debug_print_paths, ensure_data_paths
from api.monthly import router as monthly_router
from api.daily import router as daily_router


# ─────────────────────────────────────
# 1) FastAPI 앱 생성
# ─────────────────────────────────────

app = FastAPI(
    title="Breath Datalab API",
    description="월 단위 호흡기 환자율 예측 및 일 단위 위험지수 맵 시각화용 백엔드",
    version="0.1.0",
)


# ─────────────────────────────────────
# 2) CORS 설정
#    (프론트엔드 도메인에 맞게 수정 가능)
# ─────────────────────────────────────

# 개발 단계에서는 일단 모두 허용
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────
# 3) 라우터 등록
# ─────────────────────────────────────

app.include_router(monthly_router)
app.include_router(daily_router)


# ─────────────────────────────────────
# 4) 기본 헬스체크 및 정보용 엔드포인트
# ─────────────────────────────────────


@app.get("/")
def root():
    """
    간단한 헬스 체크 및 API 소개용 엔드포인트.
    """
    return {
        "message": "Breath Datalab API is running.",
        "endpoints": [
            "/api/monthly_map",
            "/api/monthly_map/available",
            "/api/daily_risk",
        ],
    }


@app.get("/api/debug/paths")
def debug_paths():
    """
    디버깅용: 현재 서버에서 인식하는 주요 경로를 로그로 찍어보고,
    클라이언트에도 간단한 정보만 리턴한다.

    실제 배포 환경에서는 필요 없으면 삭제하거나 보호해도 된다.
    """
    # 콘솔에 경로 출력
    ensure_data_paths()
    debug_print_paths()

    return {"detail": "paths printed to server log"}
