import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import toast from 'react-hot-toast'
import clsx from 'clsx'
import {
  AlertTriangle,
  BarChart2,
  CheckCircle,
  ChevronRight,
  Loader2,
  LogOut,
  Settings,
  Upload,
  User,
  Users,
} from 'lucide-react'
import { predictApi, type PredictionResponse } from '../api/client'
import { useAuth } from '../context/AuthContext'

// ── Risk badge ────────────────────────────────────────────────
function RiskBadge({ level }: { level: string }) {
  const styles = {
    high:   'bg-red-100 text-red-700 border-red-200',
    medium: 'bg-amber-100 text-amber-700 border-amber-200',
    low:    'bg-green-100 text-green-700 border-green-200',
  }[level] ?? 'bg-gray-100 text-gray-700'

  return (
    <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-semibold border ${styles}`}>
      {level.charAt(0).toUpperCase() + level.slice(1)} Risk
    </span>
  )
}

// ── Probability gauge ─────────────────────────────────────────
function ProbabilityGauge({ value }: { value: number }) {
  const pct   = Math.round(value * 100)
  const color = value >= 0.6 ? '#ef4444' : value >= 0.3 ? '#f59e0b' : '#22c55e'
  const circumference = 2 * Math.PI * 45

  return (
    <div className="flex flex-col items-center">
      <svg width="120" height="120" viewBox="0 0 120 120">
        <circle cx="60" cy="60" r="45" fill="none" stroke="#e5e7eb" strokeWidth="10" />
        <circle
          cx="60" cy="60" r="45" fill="none"
          stroke={color} strokeWidth="10"
          strokeDasharray={circumference}
          strokeDashoffset={circumference * (1 - value)}
          strokeLinecap="round"
          transform="rotate(-90 60 60)"
          style={{ transition: 'stroke-dashoffset 0.6s ease' }}
        />
        <text x="60" y="60" textAnchor="middle" dominantBaseline="central"
          fontSize="22" fontWeight="700" fill={color}>
          {pct}%
        </text>
      </svg>
      <p className="text-xs text-gray-500 mt-1">Churn Probability</p>
    </div>
  )
}

// ── Default form values ───────────────────────────────────────
const DEFAULT_FORM = {
  customer_id:       '',
  tenure:            '12',
  monthly_charges:   '65.00',
  total_charges:     '780.00',
  senior_citizen:    '0',
  partner:           'No',
  dependents:        'No',
  phone_service:     'Yes',
  multiple_lines:    'No',
  internet_service:  'Fiber optic',
  online_security:   'No',
  online_backup:     'No',
  device_protection: 'No',
  tech_support:      'No',
  streaming_tv:      'No',
  streaming_movies:  'No',
  contract:          'Month-to-month',
  paperless_billing: 'Yes',
  payment_method:    'Electronic check',
}

// ── Shared field components ───────────────────────────────────
function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="label">{label}</label>
      {children}
    </div>
  )
}

function SelectField({
  value, onChange, options,
}: { value: string; onChange: (v: string) => void; options: string[] }) {
  return (
    <select className="select" value={value} onChange={e => onChange(e.target.value)}>
      {options.map(o => <option key={o}>{o}</option>)}
    </select>
  )
}

const YES_NO = ['Yes', 'No']

// ── Main page ─────────────────────────────────────────────────
export default function PredictPage() {
  const { user, logout, isAdmin } = useAuth()
  const navigate = useNavigate()
  const [form, setForm]         = useState(DEFAULT_FORM)
  const [loading, setLoading]   = useState(false)
  const [result, setResult]     = useState<PredictionResponse | null>(null)

  const set = (key: string) => (val: string) =>
    setForm(f => ({ ...f, [key]: val }))

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setResult(null)
    try {
      const payload = {
        ...form,
        tenure:          parseInt(form.tenure),
        monthly_charges: parseFloat(form.monthly_charges),
        total_charges:   parseFloat(form.total_charges),
        senior_citizen:  parseInt(form.senior_citizen),
        customer_id:     form.customer_id || null,
      }
      const res = await predictApi.single(payload)
      setResult(res)
    } catch (err: any) {
      toast.error(err.message ?? 'Prediction failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Top nav */}
      <nav className="bg-white border-b border-gray-200 px-6 py-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 bg-brand-600 rounded-lg flex items-center justify-center">
            <BarChart2 className="w-4 h-4 text-white" />
          </div>
          <span className="font-semibold text-gray-900 text-sm">Churn Prediction</span>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => navigate('/predict')}
            className="flex items-center gap-1.5 px-3 py-1.5 text-sm text-brand-600 font-medium
                       bg-brand-50 rounded-lg"
          >
            <User className="w-3.5 h-3.5" /> Single
          </button>
          <button
            onClick={() => navigate('/batch')}
            className="flex items-center gap-1.5 px-3 py-1.5 text-sm text-gray-600
                       hover:bg-gray-100 rounded-lg transition-colors"
          >
            <Upload className="w-3.5 h-3.5" /> Batch
          </button>
          {isAdmin && (
            <button
              onClick={() => navigate('/dashboard')}
              className="flex items-center gap-1.5 px-3 py-1.5 text-sm text-gray-600
                         hover:bg-gray-100 rounded-lg transition-colors"
            >
              <Settings className="w-3.5 h-3.5" /> Dashboard
            </button>
          )}
          <div className="h-5 w-px bg-gray-200 mx-1" />
          <span className="text-xs text-gray-500 hidden sm:block">{user?.full_name}</span>
          <button onClick={() => { logout(); navigate('/login') }}
            className="p-1.5 text-gray-400 hover:text-gray-600 transition-colors" title="Sign out">
            <LogOut className="w-4 h-4" />
          </button>
        </div>
      </nav>

      <div className="max-w-6xl mx-auto px-4 py-8">
        <div className="mb-6">
          <h1 className="text-xl font-bold text-gray-900">Single Customer Prediction</h1>
          <p className="text-sm text-gray-500 mt-0.5">
            Enter customer details to get a real-time churn probability with similar customer profiles.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* ── Form ─────────────────────────────────────────── */}
          <div className="lg:col-span-2">
            <form onSubmit={handleSubmit}>
              <div className="card p-6 space-y-6">

                {/* Identity */}
                <div>
                  <h2 className="text-sm font-semibold text-gray-700 mb-4 pb-2 border-b border-gray-100">
                    Customer Identity
                  </h2>
                  <Field label="Customer ID (optional)">
                    <input className="input" placeholder="e.g. CUST-001"
                      value={form.customer_id} onChange={e => set('customer_id')(e.target.value)} />
                  </Field>
                </div>

                {/* Account */}
                <div>
                  <h2 className="text-sm font-semibold text-gray-700 mb-4 pb-2 border-b border-gray-100">
                    Account Details
                  </h2>
                  <div className="grid grid-cols-2 gap-4">
                    <Field label="Tenure (months)">
                      <input type="number" min={0} max={120} required className="input"
                        value={form.tenure} onChange={e => set('tenure')(e.target.value)} />
                    </Field>
                    <Field label="Monthly Charges ($)">
                      <input type="number" step="0.01" min={0} required className="input"
                        value={form.monthly_charges} onChange={e => set('monthly_charges')(e.target.value)} />
                    </Field>
                    <Field label="Total Charges ($)">
                      <input type="number" step="0.01" min={0} required className="input"
                        value={form.total_charges} onChange={e => set('total_charges')(e.target.value)} />
                    </Field>
                    <Field label="Senior Citizen">
                      <SelectField value={form.senior_citizen === '1' ? 'Yes' : 'No'}
                        onChange={v => set('senior_citizen')(v === 'Yes' ? '1' : '0')}
                        options={YES_NO} />
                    </Field>
                    <Field label="Partner">
                      <SelectField value={form.partner} onChange={set('partner')} options={YES_NO} />
                    </Field>
                    <Field label="Dependents">
                      <SelectField value={form.dependents} onChange={set('dependents')} options={YES_NO} />
                    </Field>
                  </div>
                </div>

                {/* Services */}
                <div>
                  <h2 className="text-sm font-semibold text-gray-700 mb-4 pb-2 border-b border-gray-100">
                    Services
                  </h2>
                  <div className="grid grid-cols-2 gap-4">
                    <Field label="Phone Service">
                      <SelectField value={form.phone_service} onChange={set('phone_service')} options={YES_NO} />
                    </Field>
                    <Field label="Multiple Lines">
                      <SelectField value={form.multiple_lines} onChange={set('multiple_lines')}
                        options={['Yes', 'No', 'No phone service']} />
                    </Field>
                    <Field label="Internet Service">
                      <SelectField value={form.internet_service} onChange={set('internet_service')}
                        options={['DSL', 'Fiber optic', 'No']} />
                    </Field>
                    <Field label="Online Security">
                      <SelectField value={form.online_security} onChange={set('online_security')}
                        options={['Yes', 'No', 'No internet service']} />
                    </Field>
                    <Field label="Online Backup">
                      <SelectField value={form.online_backup} onChange={set('online_backup')}
                        options={['Yes', 'No', 'No internet service']} />
                    </Field>
                    <Field label="Device Protection">
                      <SelectField value={form.device_protection} onChange={set('device_protection')}
                        options={['Yes', 'No', 'No internet service']} />
                    </Field>
                    <Field label="Tech Support">
                      <SelectField value={form.tech_support} onChange={set('tech_support')}
                        options={['Yes', 'No', 'No internet service']} />
                    </Field>
                    <Field label="Streaming TV">
                      <SelectField value={form.streaming_tv} onChange={set('streaming_tv')}
                        options={['Yes', 'No', 'No internet service']} />
                    </Field>
                    <Field label="Streaming Movies">
                      <SelectField value={form.streaming_movies} onChange={set('streaming_movies')}
                        options={['Yes', 'No', 'No internet service']} />
                    </Field>
                  </div>
                </div>

                {/* Billing */}
                <div>
                  <h2 className="text-sm font-semibold text-gray-700 mb-4 pb-2 border-b border-gray-100">
                    Billing
                  </h2>
                  <div className="grid grid-cols-2 gap-4">
                    <Field label="Contract">
                      <SelectField value={form.contract} onChange={set('contract')}
                        options={['Month-to-month', 'One year', 'Two year']} />
                    </Field>
                    <Field label="Paperless Billing">
                      <SelectField value={form.paperless_billing} onChange={set('paperless_billing')} options={YES_NO} />
                    </Field>
                    <Field label="Payment Method">
                      <SelectField value={form.payment_method} onChange={set('payment_method')}
                        options={['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']} />
                    </Field>
                  </div>
                </div>
              </div>

              <button type="submit" disabled={loading} className="btn-primary w-full mt-4 py-3 text-base">
                {loading
                  ? <><Loader2 className="w-5 h-5 animate-spin" /> Predicting...</>
                  : <><ChevronRight className="w-5 h-5" /> Run Prediction</>
                }
              </button>
            </form>
          </div>

          {/* ── Result panel ──────────────────────────────────── */}
          <div className="space-y-4">
            {!result && !loading && (
              <div className="card p-8 flex flex-col items-center justify-center text-center h-64">
                <BarChart2 className="w-10 h-10 text-gray-300 mb-3" />
                <p className="text-sm font-medium text-gray-500">Prediction result</p>
                <p className="text-xs text-gray-400 mt-1">Fill in the form and click Run Prediction</p>
              </div>
            )}

            {loading && (
              <div className="card p-8 flex flex-col items-center justify-center h-64">
                <Loader2 className="w-8 h-8 text-brand-600 animate-spin mb-3" />
                <p className="text-sm text-gray-500">Running ML inference...</p>
              </div>
            )}

            {result && !loading && (
              <>
                {/* Main result */}
                <div className="card p-6">
                  <div className="flex items-center justify-between mb-4">
                    <h2 className="text-sm font-semibold text-gray-700">Prediction Result</h2>
                    <RiskBadge level={result.risk_level} />
                  </div>

                  <div className="flex justify-center mb-4">
                    <ProbabilityGauge value={result.churn_probability} />
                  </div>

                  <div className="space-y-2 text-xs text-gray-500">
                    <div className="flex justify-between">
                      <span>Model version</span>
                      <span className="font-medium text-gray-700">{result.model_version}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Inference time</span>
                      <span className="font-medium text-gray-700">{result.inference_time_ms}ms</span>
                    </div>
                    {result.customer_id && (
                      <div className="flex justify-between">
                        <span>Customer ID</span>
                        <span className="font-medium text-gray-700">{result.customer_id}</span>
                      </div>
                    )}
                  </div>
                </div>

                {/* Recommendation */}
                <div className={clsx(
                  'card p-4',
                  result.risk_level === 'high'   ? 'border-red-200 bg-red-50' :
                  result.risk_level === 'medium' ? 'border-amber-200 bg-amber-50' :
                  'border-green-200 bg-green-50'
                )}>
                  <div className="flex gap-2.5">
                    {result.risk_level === 'low'
                      ? <CheckCircle className="w-4 h-4 text-green-600 flex-shrink-0 mt-0.5" />
                      : <AlertTriangle className={clsx(
                          'w-4 h-4 flex-shrink-0 mt-0.5',
                          result.risk_level === 'high' ? 'text-red-600' : 'text-amber-600'
                        )} />
                    }
                    <p className="text-xs leading-relaxed text-gray-700">{result.recommendation}</p>
                  </div>
                </div>

                {/* Similar customers */}
                {result.similar_customers.length > 0 && (
                  <div className="card p-4">
                    <div className="flex items-center gap-2 mb-3">
                      <Users className="w-4 h-4 text-gray-500" />
                      <h3 className="text-sm font-semibold text-gray-700">Similar Customers</h3>
                    </div>
                    <p className="text-xs text-gray-400 mb-3">
                      Top {result.similar_customers.length} nearest neighbours from training data (FAISS)
                    </p>
                    <div className="space-y-2">
                      {result.similar_customers.map((s, i) => (
                        <div key={i} className="flex items-center gap-2 text-xs">
                          <div className={clsx(
                            'w-2 h-2 rounded-full flex-shrink-0',
                            s.churn_label === 1 ? 'bg-red-400' : 'bg-green-400'
                          )} />
                          <span className="text-gray-600 flex-1">
                            {s.churn_label === 1 ? 'Churned' : 'Retained'}
                          </span>
                          <div className="flex-1 bg-gray-100 rounded-full h-1.5">
                            <div
                              className="bg-brand-500 h-1.5 rounded-full"
                              style={{ width: `${Math.round(s.similarity_score * 100)}%` }}
                            />
                          </div>
                          <span className="text-gray-500 w-8 text-right">
                            {Math.round(s.similarity_score * 100)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
