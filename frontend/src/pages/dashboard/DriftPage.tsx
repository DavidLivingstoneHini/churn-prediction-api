import { useState } from 'react'
import toast from 'react-hot-toast'
import clsx from 'clsx'
import { AlertTriangle, CheckCircle, GitBranch, Info, Loader2, RefreshCw } from 'lucide-react'
import { adminApi, type DriftReport } from '../../api/client'

function PsiBar({ value, threshold = 0.2 }: { value: number; threshold?: number }) {
  const pct   = Math.min(value / 0.4, 1) * 100
  const color = value >= threshold ? 'bg-red-500' : value >= 0.1 ? 'bg-amber-500' : 'bg-green-500'
  return (
    <div className="flex items-center gap-3">
      <div className="flex-1 bg-gray-100 rounded-full h-2">
        <div className={`${color} h-2 rounded-full transition-all`} style={{ width: `${pct}%` }} />
      </div>
      <span className={clsx(
        'text-xs font-semibold w-12 text-right',
        value >= threshold ? 'text-red-600' : value >= 0.1 ? 'text-amber-600' : 'text-green-600'
      )}>
        {value.toFixed(4)}
      </span>
    </div>
  )
}

export default function DriftPage() {
  const [report,  setReport]  = useState<DriftReport | null>(null)
  const [loading, setLoading] = useState(false)

  const runCheck = async () => {
    setLoading(true)
    try {
      const r = await adminApi.getDrift()
      setReport(r)
      if (r.drift_detected) {
        toast.error(`Drift detected in ${Object.keys(r.drifted_features).length} feature(s)`)
      } else {
        toast.success('No significant drift detected')
      }
    } catch (err: any) {
      toast.error(err.message ?? 'Drift check failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-gray-900">Data Drift Monitor</h1>
          <p className="text-sm text-gray-500 mt-0.5">
            PSI-based drift detection comparing training vs. recent inference distribution.
          </p>
        </div>
        <button onClick={runCheck} disabled={loading} className="btn-primary">
          {loading
            ? <><Loader2 className="w-4 h-4 animate-spin" />Running...</>
            : <><RefreshCw className="w-4 h-4" />Run drift check</>
          }
        </button>
      </div>

      {/* PSI explanation */}
      <div className="card p-4 bg-blue-50 border-blue-200">
        <div className="flex gap-2.5">
          <Info className="w-4 h-4 text-blue-600 flex-shrink-0 mt-0.5" />
          <div className="text-xs text-blue-700 space-y-0.5">
            <p><strong>Population Stability Index (PSI)</strong> measures how much a feature's distribution has shifted.</p>
            <p><span className="text-green-700 font-semibold">PSI &lt; 0.1</span> — Stable &nbsp;
               <span className="text-amber-700 font-semibold">PSI 0.1–0.2</span> — Monitor &nbsp;
               <span className="text-red-700 font-semibold">PSI ≥ 0.2</span> — Significant drift → retrain
            </p>
          </div>
        </div>
      </div>

      {!report && !loading && (
        <div className="card p-12 flex flex-col items-center justify-center text-center">
          <GitBranch className="w-10 h-10 text-gray-300 mb-3" />
          <p className="text-sm font-medium text-gray-500">No drift report yet</p>
          <p className="text-xs text-gray-400 mt-1">
            Click "Run drift check" to compare training vs. recent prediction distributions.
            Requires ≥ 30 recent predictions.
          </p>
        </div>
      )}

      {report && (
        <div className="space-y-4">
          {/* Overall status */}
          <div className={clsx(
            'card p-5 flex items-center gap-4',
            report.drift_detected ? 'border-red-200 bg-red-50' : 'border-green-200 bg-green-50'
          )}>
            {report.drift_detected
              ? <AlertTriangle className="w-8 h-8 text-red-600 flex-shrink-0" />
              : <CheckCircle   className="w-8 h-8 text-green-600 flex-shrink-0" />
            }
            <div className="flex-1">
              <p className={clsx('font-semibold', report.drift_detected ? 'text-red-700' : 'text-green-700')}>
                {report.drift_detected ? 'Drift detected — retraining recommended' : 'No significant drift detected'}
              </p>
              <p className="text-xs text-gray-600 mt-0.5">
                Max PSI: {report.max_psi.toFixed(4)} &nbsp;·&nbsp;
                Avg PSI: {report.avg_psi.toFixed(4)} &nbsp;·&nbsp;
                Threshold: {report.threshold} &nbsp;·&nbsp;
                Reference: {report.n_reference.toLocaleString()} rows &nbsp;·&nbsp;
                Current: {report.n_current} rows
              </p>
            </div>
            <p className="text-xs text-gray-400 flex-shrink-0">
              {new Date(report.checked_at).toLocaleString()}
            </p>
          </div>

          {/* Drifted features */}
          {Object.keys(report.drifted_features).length > 0 && (
            <div className="card p-5">
              <h2 className="text-sm font-semibold text-red-700 mb-3 flex items-center gap-1.5">
                <AlertTriangle className="w-4 h-4" />
                Drifted features ({Object.keys(report.drifted_features).length})
              </h2>
              <div className="space-y-3">
                {Object.entries(report.drifted_features).map(([feat, psi]) => (
                  <div key={feat}>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="font-medium text-gray-700">{feat}</span>
                      <span className="text-red-600 font-semibold">PSI {psi.toFixed(4)}</span>
                    </div>
                    <PsiBar value={psi} />
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Monitored features */}
          {Object.keys(report.monitored_features).length > 0 && (
            <div className="card p-5">
              <h2 className="text-sm font-semibold text-amber-700 mb-3">
                Monitored features ({Object.keys(report.monitored_features).length})
              </h2>
              <div className="space-y-3">
                {Object.entries(report.monitored_features).map(([feat, psi]) => (
                  <div key={feat}>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="font-medium text-gray-700">{feat}</span>
                      <span className="text-amber-600 font-semibold">PSI {psi.toFixed(4)}</span>
                    </div>
                    <PsiBar value={psi} />
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Stable count */}
          <div className="card p-4 flex items-center gap-3">
            <CheckCircle className="w-5 h-5 text-green-600 flex-shrink-0" />
            <p className="text-sm text-gray-700">
              <strong>{report.stable_feature_count}</strong> features are stable (PSI &lt; 0.1)
            </p>
          </div>
        </div>
      )}
    </div>
  )
}
