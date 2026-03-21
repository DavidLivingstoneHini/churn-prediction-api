import { useEffect, useState } from 'react'
import {
  AreaChart, Area, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts'
import { Activity, AlertTriangle, CheckCircle, Clock, TrendingDown } from 'lucide-react'
import toast from 'react-hot-toast'
import { adminApi, type Analytics, type ModelInfo } from '../../api/client'

const RISK_COLORS = ['#ef4444', '#f59e0b', '#22c55e']

function StatCard({ label, value, sub, icon: Icon, cls }: {
  label: string; value: string | number; sub?: string
  icon: React.ElementType; cls: string
}) {
  return (
    <div className="card p-5">
      <div className="flex items-center justify-between mb-3">
        <p className="text-sm font-medium text-gray-500">{label}</p>
        <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${cls}`}>
          <Icon className="w-4 h-4" />
        </div>
      </div>
      <p className="text-2xl font-bold text-gray-900">{value}</p>
      {sub && <p className="text-xs text-gray-400 mt-0.5">{sub}</p>}
    </div>
  )
}

function ModelBadge({ info }: { info: ModelInfo }) {
  return (
    <div className="card p-4">
      <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-3">Active Model</p>
      <div className="grid grid-cols-2 gap-x-6 gap-y-2">
        {[
          ['Version',   info.version],
          ['AUC-ROC',   info.auc_roc.toFixed(4)],
          ['Precision', info.precision.toFixed(4)],
          ['Recall',    info.recall.toFixed(4)],
          ['F1 Score',  info.f1_score.toFixed(4)],
          ['Train rows', info.training_rows.toLocaleString()],
        ].map(([k, v]) => (
          <div key={k}>
            <p className="text-xs text-gray-400">{k}</p>
            <p className="text-sm font-semibold text-gray-800">{v}</p>
          </div>
        ))}
      </div>
    </div>
  )
}

export default function AnalyticsPage() {
  const [analytics, setAnalytics] = useState<Analytics | null>(null)
  const [model,     setModel]     = useState<ModelInfo | null>(null)
  const [loading,   setLoading]   = useState(true)

  useEffect(() => {
    Promise.all([adminApi.getAnalytics(), adminApi.getModel()])
      .then(([a, m]) => { setAnalytics(a); setModel(m) })
      .catch(() => toast.error('Failed to load analytics'))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return (
    <div className="flex items-center justify-center h-full">
      <div className="w-7 h-7 border-2 border-brand-600 border-t-transparent rounded-full animate-spin" />
    </div>
  )
  if (!analytics) return null

  return (
    <div className="p-6 space-y-6">
      <div>
        <h1 className="text-xl font-bold text-gray-900">Analytics</h1>
        <p className="text-sm text-gray-500 mt-0.5">Prediction performance and volume overview</p>
      </div>

      {/* Stat cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard label="Total predictions" value={analytics.total_predictions.toLocaleString()}
          sub={`${analytics.predictions_today} today`}
          icon={Activity} cls="bg-blue-100 text-blue-600" />
        <StatCard label="High risk customers" value={analytics.high_risk_count.toLocaleString()}
          sub="Immediate action needed"
          icon={AlertTriangle} cls="bg-red-100 text-red-600" />
        <StatCard label="Avg churn probability" value={`${Math.round(analytics.avg_churn_probability * 100)}%`}
          sub="Across all predictions"
          icon={TrendingDown} cls="bg-amber-100 text-amber-600" />
        <StatCard label="Avg inference time" value={`${analytics.avg_inference_time_ms.toFixed(0)}ms`}
          sub="Per prediction"
          icon={Clock} cls="bg-purple-100 text-purple-600" />
      </div>

      {/* Charts row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Volume chart */}
        <div className="card p-5 lg:col-span-2">
          <h2 className="text-sm font-semibold text-gray-700 mb-4">Prediction volume — last 14 days</h2>
          <ResponsiveContainer width="100%" height={220}>
            <AreaChart data={analytics.daily_volume} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
              <defs>
                <linearGradient id="volGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"  stopColor="#22c55e" stopOpacity={0.15} />
                  <stop offset="95%" stopColor="#22c55e" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="date" tick={{ fontSize: 11 }} tickLine={false} axisLine={false} />
              <YAxis tick={{ fontSize: 11 }} tickLine={false} axisLine={false} allowDecimals={false} />
              <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8, border: '1px solid #e5e7eb' }} />
              <Area type="monotone" dataKey="predictions" stroke="#22c55e" strokeWidth={2}
                fill="url(#volGrad)" dot={false} activeDot={{ r: 4 }} />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Risk pie */}
        <div className="card p-5">
          <h2 className="text-sm font-semibold text-gray-700 mb-4">Risk distribution</h2>
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie data={analytics.risk_distribution} cx="50%" cy="45%"
                innerRadius={55} outerRadius={80} paddingAngle={3} dataKey="count">
                {analytics.risk_distribution.map((_, i) => (
                  <Cell key={i} fill={RISK_COLORS[i % RISK_COLORS.length]} />
                ))}
              </Pie>
              <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8 }} />
              <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 12 }} />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Histogram + model info */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="card p-5 lg:col-span-2">
          <h2 className="text-sm font-semibold text-gray-700 mb-4">Churn probability distribution</h2>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={analytics.churn_probability_histogram} margin={{ top: 0, right: 10, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="range" tick={{ fontSize: 10 }} tickLine={false} axisLine={false} />
              <YAxis tick={{ fontSize: 11 }} tickLine={false} axisLine={false} allowDecimals={false} />
              <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8 }} />
              <Bar dataKey="count" radius={[3, 3, 0, 0]}
                fill="#22c55e"
                label={false}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {model && <ModelBadge info={model} />}
      </div>
    </div>
  )
}
