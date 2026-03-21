import { useEffect, useState } from 'react'
import toast from 'react-hot-toast'
import clsx from 'clsx'
import { ChevronLeft, ChevronRight, Clock, Loader2 } from 'lucide-react'
import { adminApi } from '../../api/client'

const PAGE_SIZE = 20

function RiskChip({ level }: { level: string }) {
  return (
    <span className={clsx(
      'inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium',
      level === 'high'   ? 'bg-red-100 text-red-700' :
      level === 'medium' ? 'bg-amber-100 text-amber-700' :
                           'bg-green-100 text-green-700'
    )}>
      {level.charAt(0).toUpperCase() + level.slice(1)}
    </span>
  )
}

function ProbBar({ value }: { value: number }) {
  const color = value >= 0.6 ? 'bg-red-500' : value >= 0.3 ? 'bg-amber-500' : 'bg-green-500'
  return (
    <div className="flex items-center gap-2">
      <div className="w-20 bg-gray-100 rounded-full h-1.5">
        <div className={`${color} h-1.5 rounded-full`} style={{ width: `${Math.round(value * 100)}%` }} />
      </div>
      <span className="text-xs text-gray-600 w-8">{Math.round(value * 100)}%</span>
    </div>
  )
}

export default function HistoryPage() {
  const [items,   setItems]   = useState<any[]>([])
  const [total,   setTotal]   = useState(0)
  const [page,    setPage]    = useState(0)
  const [loading, setLoading] = useState(true)

  const fetchPage = async (p: number) => {
    setLoading(true)
    try {
      const data = await adminApi.getHistory(PAGE_SIZE, p * PAGE_SIZE)
      setItems(data.items)
      setTotal(data.total)
    } catch {
      toast.error('Failed to load history')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetchPage(0) }, [])

  const totalPages = Math.ceil(total / PAGE_SIZE)

  return (
    <div className="p-6 space-y-6">
      <div>
        <h1 className="text-xl font-bold text-gray-900">Prediction Audit Log</h1>
        <p className="text-sm text-gray-500 mt-0.5">
          Full audit trail of all predictions — {total.toLocaleString()} total records.
        </p>
      </div>

      <div className="card overflow-hidden">
        {/* Header */}
        <div className="grid grid-cols-12 gap-4 px-5 py-3 bg-gray-50 border-b border-gray-200
                        text-xs font-semibold text-gray-500 uppercase tracking-wide">
          <div className="col-span-3">Customer ID</div>
          <div className="col-span-2">Risk Level</div>
          <div className="col-span-3">Churn Probability</div>
          <div className="col-span-2">Inference</div>
          <div className="col-span-2">Timestamp</div>
        </div>

        {/* Rows */}
        {loading ? (
          <div className="flex items-center justify-center py-14">
            <Loader2 className="w-6 h-6 text-gray-400 animate-spin" />
          </div>
        ) : items.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-14 text-center">
            <Clock className="w-8 h-8 text-gray-300 mb-2" />
            <p className="text-sm text-gray-500">No predictions recorded yet</p>
          </div>
        ) : (
          <div className="divide-y divide-gray-100">
            {items.map(item => (
              <div key={item.id}
                className="grid grid-cols-12 gap-4 px-5 py-3 hover:bg-gray-50 transition-colors text-sm">
                <div className="col-span-3 text-gray-600 truncate font-mono text-xs">
                  {item.customer_id || <span className="text-gray-400 italic">—</span>}
                </div>
                <div className="col-span-2">
                  <RiskChip level={item.risk_level} />
                </div>
                <div className="col-span-3">
                  <ProbBar value={item.churn_probability} />
                </div>
                <div className="col-span-2 text-xs text-gray-500">
                  {item.inference_time_ms}ms
                </div>
                <div className="col-span-2 text-xs text-gray-400">
                  {new Date(item.created_at).toLocaleDateString('en-GB', {
                    day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit'
                  })}
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex items-center justify-between px-5 py-3 border-t border-gray-200 bg-gray-50">
            <p className="text-xs text-gray-500">
              Showing {page * PAGE_SIZE + 1}–{Math.min((page + 1) * PAGE_SIZE, total)} of {total.toLocaleString()}
            </p>
            <div className="flex items-center gap-1">
              <button
                onClick={() => { setPage(p => p - 1); fetchPage(page - 1) }}
                disabled={page === 0 || loading}
                className="p-1.5 text-gray-500 hover:text-gray-700 disabled:opacity-40 disabled:cursor-not-allowed"
              >
                <ChevronLeft className="w-4 h-4" />
              </button>
              <span className="text-xs text-gray-600 px-2">
                Page {page + 1} of {totalPages}
              </span>
              <button
                onClick={() => { setPage(p => p + 1); fetchPage(page + 1) }}
                disabled={page >= totalPages - 1 || loading}
                className="p-1.5 text-gray-500 hover:text-gray-700 disabled:opacity-40 disabled:cursor-not-allowed"
              >
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
