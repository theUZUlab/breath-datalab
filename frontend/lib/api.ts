const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? 'http://localhost:8000';

export type MetricType = 'actual' | 'pred' | 'error_pct';

export interface YearMonth {
    year: number;
    month: number;
}

export interface MonthlyMapRowExtra {
    actual: number;
    pred: number;
    error_pct: number;
    year: number;
    month: number;
    ym: string;
}

export interface MonthlyMapRow {
    sido_name: string;
    value: number;
    extra: MonthlyMapRowExtra;
}

export interface MonthlyMapResponse {
    ym: string;
    year: number;
    month: number;
    metric: MetricType;
    data: MonthlyMapRow[];
}

export interface MonthlyAvailableResponse {
    items: YearMonth[];
}

export async function fetchAvailableMonths(): Promise<YearMonth[]> {
    const res = await fetch(`${API_BASE_URL}/api/monthly_map/available`);
    if (!res.ok) {
        throw new Error('사용 가능한 연월 목록을 불러오지 못했습니다.');
    }
    const json: MonthlyAvailableResponse = await res.json();
    return json.items;
}

export async function fetchMonthlyMap(ym: string, metric: MetricType = 'actual'): Promise<MonthlyMapResponse> {
    const url = new URL(`${API_BASE_URL}/api/monthly_map`);
    url.searchParams.set('ym', ym);
    url.searchParams.set('metric', metric);

    const res = await fetch(url.toString());
    if (!res.ok) {
        const text = await res.text();
        throw new Error(`월 맵 데이터를 불러오지 못했습니다. status=${res.status}, body=${text}`);
    }

    const json = (await res.json()) as MonthlyMapResponse;
    return json;
}

/**
 * 일 단위 위험지수 타입들
 */

export interface DailyRiskApiItem {
    sido_name: string;
    value: number;
    extra?: {
        risk_score?: number;
        risk_level?: string;
        date?: string;
        [key: string]: unknown;
    };
}

export interface DailyRiskApiResponse {
    date: string;
    env_used: boolean;
    data: DailyRiskApiItem[];
}

export interface DailyRiskRow {
    sido_name: string;
    risk_score: number;
    risk_level: string;
    date: string;
}

/**
 * GET /api/daily_risk?date=YYYY-MM-DD 호출 헬퍼
 */
export async function fetchDailyRisk(date: string): Promise<DailyRiskRow[]> {
    const url = new URL(`${API_BASE_URL}/api/daily_risk`);
    url.searchParams.set('date', date);

    const res = await fetch(url.toString());
    if (!res.ok) {
        const text = await res.text();
        throw new Error(`일 단위 위험지수를 불러오지 못했습니다. status=${res.status}, body=${text}`);
    }

    const json = (await res.json()) as DailyRiskApiResponse;

    // API 응답 예시:
    // {
    //   "date": "2024-01-01",
    //   "env_used": true,
    //   "data": [
    //     {
    //       "sido_name": "서울",
    //       "value": 78.5,
    //       "extra": {
    //         "risk_score": 78.5,
    //         "risk_level": "High",
    //         "date": "2024-01-01"
    //       }
    //     },
    //     ...
    //   ]
    // }

    return (json.data ?? []).map((item) => {
        const riskScore = item.extra?.risk_score ?? item.value;
        const riskLevel = item.extra?.risk_level ?? '';
        const rowDate = item.extra?.date ?? json.date;

        return {
            sido_name: item.sido_name,
            risk_score: riskScore,
            risk_level: riskLevel,
            date: rowDate,
        };
    });
}
