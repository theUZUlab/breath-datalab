// frontend/app/page.tsx
'use client';

import { useEffect, useMemo, useState } from 'react';
import { fetchAvailableMonths, fetchMonthlyMap, MetricType, MonthlyMapRow, YearMonth } from '@/lib/api';

type TabType = 'monthly' | 'daily';

function formatYm(year: number, month: number) {
    return `${year}-${String(month).padStart(2, '0')}`;
}

function metricLabel(metric: MetricType) {
    switch (metric) {
        case 'actual':
            return '실제 환자율 (resp_rate_total)';
        case 'pred':
            return '예측 환자율 (y_hat)';
        case 'error_pct':
            return '오차율 (%)';
    }
}

export default function HomePage() {
    const [tab, setTab] = useState<TabType>('monthly');

    const [availableMonths, setAvailableMonths] = useState<YearMonth[]>([]);
    const [selectedYm, setSelectedYm] = useState<string | null>(null);
    const [metric, setMetric] = useState<MetricType>('actual');

    const [rows, setRows] = useState<MonthlyMapRow[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // 1) 초기 로딩: 사용 가능한 연월 가져오기
    useEffect(() => {
        async function loadAvailableMonths() {
            try {
                setError(null);
                const items = await fetchAvailableMonths();
                setAvailableMonths(items);
                if (items.length > 0) {
                    const last = items[items.length - 1];
                    setSelectedYm(formatYm(last.year, last.month));
                }
            } catch (e) {
                console.error(e);
                const message = e instanceof Error ? e.message : '연월 목록을 불러오는 중 오류가 발생했습니다.';
                setError(message);
            }
        }

        loadAvailableMonths();
    }, []);

    // 2) 선택된 ym / metric 바뀔 때마다 월 맵 데이터 로딩
    useEffect(() => {
        async function loadMonthlyMap() {
            if (!selectedYm) return;
            try {
                setLoading(true);
                setError(null);
                const res = await fetchMonthlyMap(selectedYm, metric);
                setRows(res.data);
            } catch (e) {
                console.error(e);
                const message = e instanceof Error ? e.message : '월 맵 데이터를 불러오는 중 오류가 발생했습니다.';
                setError(message);
                setRows([]);
            } finally {
                setLoading(false);
            }
        }
        loadMonthlyMap();
    }, [selectedYm, metric]);

    const monthlyOptions = useMemo(
        () =>
            availableMonths.map((ym) => ({
                value: formatYm(ym.year, ym.month),
                label: `${ym.year}년 ${ym.month}월`,
            })),
        [availableMonths]
    );

    const currentYmLabel = useMemo(() => {
        if (!selectedYm) return '';
        const [y, m] = selectedYm.split('-');
        return `${y}년 ${parseInt(m, 10)}월`;
    }, [selectedYm]);

    return (
        <main className="min-h-screen px-6 py-8">
            <h1 className="text-2xl font-bold mb-4">Breath Datalab 대시보드 (MVP)</h1>

            {/* 탭 */}
            <div className="flex gap-2 mb-6">
                <button
                    type="button"
                    onClick={() => setTab('monthly')}
                    className={`px-3 py-2 border rounded ${tab === 'monthly' ? 'bg-black text-white' : 'bg-white'}`}
                >
                    월 단위 맵
                </button>
                <button
                    type="button"
                    onClick={() => setTab('daily')}
                    className={`px-3 py-2 border rounded ${tab === 'daily' ? 'bg-gray-200 text-gray-500' : 'bg-white'}`}
                    disabled
                >
                    일 단위 위험지수 (준비 중)
                </button>
            </div>

            {tab === 'monthly' && (
                <section className="space-y-4">
                    {/* 컨트롤 영역 */}
                    <div className="flex flex-wrap gap-4 items-center">
                        {/* 연월 선택 */}
                        <label className="flex items-center gap-2">
                            <span>연월 선택:</span>
                            <select
                                value={selectedYm ?? ''}
                                onChange={(e) => setSelectedYm(e.target.value)}
                                className="border px-2 py-1 rounded"
                            >
                                {monthlyOptions.length === 0 && <option value="">연월 없음</option>}
                                {monthlyOptions.map((opt) => (
                                    <option key={opt.value} value={opt.value}>
                                        {opt.label}
                                    </option>
                                ))}
                            </select>
                        </label>

                        {/* metric 선택 */}
                        <label className="flex items-center gap-2">
                            <span>지표 선택:</span>
                            <select
                                value={metric}
                                onChange={(e) => setMetric(e.target.value as MetricType)}
                                className="border px-2 py-1 rounded"
                            >
                                <option value="actual">실제 환자율 (actual)</option>
                                <option value="pred">예측 환자율 (pred)</option>
                                <option value="error_pct">오차율 (error_pct)</option>
                            </select>
                        </label>
                    </div>

                    {/* 상태 표시 */}
                    {loading && <p>로딩 중입니다...</p>}
                    {error && <p style={{ color: 'red' }}>오류: {error}</p>}

                    {/* 테이블 */}
                    {!loading && !error && rows.length > 0 && (
                        <div className="mt-4 overflow-x-auto">
                            <h2 className="text-lg font-semibold mb-2">
                                {currentYmLabel} · {metricLabel(metric)}
                            </h2>
                            <table className="min-w-[480px] border-collapse border">
                                <thead>
                                    <tr>
                                        <th className="border px-2 py-1">시/도</th>
                                        <th className="border px-2 py-1">value (색칠용)</th>
                                        <th className="border px-2 py-1">actual</th>
                                        <th className="border px-2 py-1">pred</th>
                                        <th className="border px-2 py-1">error_pct</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {rows.map((row) => (
                                        <tr key={row.sido_name}>
                                            <td className="border px-2 py-1">{row.sido_name}</td>
                                            <td className="border px-2 py-1">{row.value.toFixed(3)}</td>
                                            <td className="border px-2 py-1">{row.extra.actual.toFixed(3)}</td>
                                            <td className="border px-2 py-1">{row.extra.pred.toFixed(3)}</td>
                                            <td className="border px-2 py-1">{row.extra.error_pct.toFixed(3)}</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    )}

                    {!loading && !error && rows.length === 0 && <p>표시할 데이터가 없습니다.</p>}
                </section>
            )}

            {tab === 'daily' && (
                <section>
                    <p>일 단위 위험지수 탭은 다음 브랜치에서 구현 예정입니다.</p>
                </section>
            )}
        </main>
    );
}
