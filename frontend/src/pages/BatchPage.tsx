import { useCallback, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useDropzone } from 'react-dropzone'
import toast from 'react-hot-toast'
import clsx from 'clsx'
import {
  BarChart2, Download, Loader2, LogOut, Settings,
  Upload, User, FileText, AlertTriangle, CheckCircle, TrendingUp,
} from 'lucide-react'
import { predictApi, type BatchSummary } from '../api/client'
import { useAuth } from '../context/AuthContext'

function StatCard({ label, value, sub, color }: {
  label: string; value: string | number; sub?: string; color: string
}) {
  return (
    <div className={`card p-4 border-l-4 ${color}`}>
      <p className="text-xs font-medium text-gray-500">{label}</p>
      <p className="text-2xl font-bold text-gray-900 mt-0.5">{value}</p>
      {sub && <p className="text-xs text-gray-400 mt-0.5">{sub}</p>}
    </div>
  )
}

const SAMPLE_CSV = `customerID,tenure,MonthlyCharges,TotalCharges,Contract,InternetService,PaymentMethod,Partner,SeniorCitizen
CUST-001,12,65.00,780.00,Month-to-month,Fiber optic,Electronic check,No,0
CUST-002,48,45.50,2184.00,One year,DSL,Bank transfer (automatic),Yes,0
CUST-003,2,89.00,178.00,Month-to-month,Fiber optic,Electronic check,No,1
CUST-004,72,20.00,1440.00,Two year,DSL,Mailed check,Yes,0
CUST-005,6,75.00,450.00,Month-to-month,Fiber optic,Credit card (automatic),No,0`

export default function BatchPage() {
  const { user, logout, isAdmin } = useAuth()
  const navigate = useNavigate()
  const [loading, setLoading] = useState(false)
  const [result, setResult]   = useState<BatchSummary | null>(null)

  const onDrop = useCallback(async (accepted: File[]) => {
    if (!accepted.length) return
    const file = accepted[0]
    setLoading(true)
    setResult(null)
    toast.loading(`Processing ${file.name}...`, { id: 'batch' })
    try {
      const res = await predictApi.batch(file)
      setResult(res)
      toast.success(`Processed ${res.total_rows} customers`, { id: 'batch' })
    } catch (err: any) {
      toast.error(err.message ?? 'Batch prediction failed', { id: 'batch' })
    } finally {
      setLoading(false)
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'text/csv': ['.csv'] },
    maxSize: 10 * 1024 * 1024,
    multiple: false,
    onDropRejected: r => toast.error(r[0]?.errors[0]?.message ?? 'Invalid file'),
  })

  const downloadSample = () => {
    const blob = new Blob([SAMPLE_CSV], { type: 'text/csv' })
    const url  = URL.createObjectURL(blob)
    const a    = document.createElement('a')
    a.href     = url
    a.download = 'sample_customers.csv'
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Nav */}
      <nav className="bg-white border-b border-gray-200 px-6 py-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 bg-brand-600 rounded-lg flex items-center justify-center">
            <BarChart2 className="w-4 h-4 text-white" />
          </div>
          <span className="font-semibold text-gray-900 text-sm">Churn Prediction</span>
        </div>
        <div className="flex items-center gap-2">
          <button onClick={() => navigate('/predict')}
            className="flex items-center gap-1.5 px-3 py-1.5 text-sm text-gray-600 hover:bg-gray-100 rounded-lg transition-colors">
            <User className="w-3.5 h-3.5" /> Single
          </button>
          <button onClick={() => navigate('/batch')}
            className="flex items-center gap-1.5 px-3 py-1.5 text-sm text-brand-600 font-medium bg-brand-50 rounded-lg">
            <Upload className="w-3.5 h-3.5" /> Batch
          </button>
          {isAdmin && (
            <button onClick={() => navigate('/dashboard')}
              className="flex items-center gap-1.5 px-3 py-1.5 text-sm text-gray-600 hover:bg-gray-100 rounded-lg transition-colors">
              <Settings className="w-3.5 h-3.5" /> Dashboard
            </button>
          )}
          <div className="h-5 w-px bg-gray-200 mx-1" />
          <button onClick={() => { logout(); navigate('/login') }}
            className="p-1.5 text-gray-400 hover:text-gray-600 transition-colors">
            <LogOut className="w-4 h-4" />
          </button>
        </div>
      </nav>

      <div className="max-w-4xl mx-auto px-4 py-8 space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold text-gray-900">Batch Prediction</h1>
            <p className="text-sm text-gray-500 mt-0.5">
              Upload a CSV file to predict churn for multiple customers at once.
            </p>
          </div>
          <button onClick={downloadSample} className="btn-secondary text-xs">
            <Download className="w-3.5 h-3.5" />
            Download sample CSV
          </button>
        </div>

        {/* Required columns info */}
        <div className="card p-4 bg-blue-50 border-blue-200">
          <p className="text-xs font-semibold text-blue-700 mb-1">Required CSV columns:</p>
          <p className="text-xs text-blue-600 font-mono">
            tenure, MonthlyCharges, TotalCharges, Contract, InternetService, PaymentMethod
          </p>
          <p className="text-xs text-blue-500 mt-1">
            Optional: customerID, Partner, Dependents, PhoneService, SeniorCitizen, and all service columns
          </p>
        </div>

        {/* Drop zone */}
        <div
          {...getRootProps()}
          className={clsx(
            'border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-colors',
            isDragActive
              ? 'border-brand-500 bg-brand-50'
              : 'border-gray-300 bg-white hover:border-brand-400 hover:bg-gray-50'
          )}
        >
          <input {...getInputProps()} />
          <div className="flex flex-col items-center gap-3">
            <div className={clsx(
              'w-14 h-14 rounded-xl flex items-center justify-center transition-colors',
              isDragActive ? 'bg-brand-100' : 'bg-gray-100'
            )}>
              {loading
                ? <Loader2 className="w-7 h-7 text-brand-600 animate-spin" />
                : <FileText className={clsx('w-7 h-7', isDragActive ? 'text-brand-600' : 'text-gray-500')} />
              }
            </div>
            <div>
              <p className="text-sm font-medium text-gray-700">
                {isDragActive ? 'Drop your CSV here' : 'Drag & drop a CSV file, or click to browse'}
              </p>
              <p className="text-xs text-gray-400 mt-0.5">Max 10 MB · CSV only</p>
            </div>
          </div>
        </div>

        {/* Results */}
        {result && (
          <div className="space-y-4">
            <div className="flex items-center gap-2">
              <CheckCircle className="w-5 h-5 text-green-600" />
              <h2 className="text-base font-semibold text-gray-900">
                Batch complete — {result.filename}
              </h2>
            </div>

            {/* Stat cards */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <StatCard label="Total customers" value={result.total_rows}
                color="border-gray-400" />
              <StatCard label="High risk" value={result.high_risk}
                sub={`${Math.round(result.high_risk / result.total_rows * 100)}% of total`}
                color="border-red-400" />
              <StatCard label="Medium risk" value={result.medium_risk}
                sub={`${Math.round(result.medium_risk / result.total_rows * 100)}% of total`}
                color="border-amber-400" />
              <StatCard label="Low risk" value={result.low_risk}
                sub={`${Math.round(result.low_risk / result.total_rows * 100)}% of total`}
                color="border-green-400" />
            </div>

            {/* Summary card */}
            <div className="card p-5">
              <div className="flex items-center gap-2 mb-4">
                <TrendingUp className="w-4 h-4 text-gray-500" />
                <h3 className="text-sm font-semibold text-gray-700">Batch Summary</h3>
              </div>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <p className="text-gray-500 text-xs">Avg churn probability</p>
                  <p className="font-bold text-2xl text-gray-900 mt-0.5">
                    {Math.round(result.avg_churn_probability * 100)}%
                  </p>
                </div>
                <div>
                  <p className="text-gray-500 text-xs">Batch status</p>
                  <p className="font-semibold text-gray-900 mt-0.5 capitalize">{result.status}</p>
                </div>
              </div>

              {/* Risk bar */}
              <div className="mt-4">
                <p className="text-xs text-gray-500 mb-1.5">Risk distribution</p>
                <div className="flex h-4 rounded-full overflow-hidden gap-0.5">
                  {result.high_risk > 0 && (
                    <div
                      className="bg-red-400 flex items-center justify-center text-white text-xs font-medium"
                      style={{ width: `${result.high_risk / result.total_rows * 100}%` }}
                    >
                      {result.high_risk > result.total_rows * 0.05 ? `${Math.round(result.high_risk / result.total_rows * 100)}%` : ''}
                    </div>
                  )}
                  {result.medium_risk > 0 && (
                    <div
                      className="bg-amber-400 flex items-center justify-center text-white text-xs font-medium"
                      style={{ width: `${result.medium_risk / result.total_rows * 100}%` }}
                    >
                      {result.medium_risk > result.total_rows * 0.05 ? `${Math.round(result.medium_risk / result.total_rows * 100)}%` : ''}
                    </div>
                  )}
                  {result.low_risk > 0 && (
                    <div
                      className="bg-green-400 flex-1 flex items-center justify-center text-white text-xs font-medium"
                    >
                      {Math.round(result.low_risk / result.total_rows * 100)}%
                    </div>
                  )}
                </div>
                <div className="flex gap-4 mt-2 text-xs text-gray-500">
                  <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-red-400" />High</span>
                  <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-amber-400" />Medium</span>
                  <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-green-400" />Low</span>
                </div>
              </div>

              {result.high_risk > 0 && (
                <div className="mt-4 flex items-start gap-2 p-3 bg-red-50 border border-red-200 rounded-lg">
                  <AlertTriangle className="w-4 h-4 text-red-600 flex-shrink-0 mt-0.5" />
                  <p className="text-xs text-red-700">
                    <strong>{result.high_risk} customers</strong> are at high churn risk.
                    Immediate retention campaigns are recommended for this cohort.
                  </p>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
