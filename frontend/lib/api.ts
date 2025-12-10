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
