const API = '/api/v1'

export const tokenStorage = {
  getAccess:  (): string | null => localStorage.getItem('access_token'),
  getRefresh: (): string | null => localStorage.getItem('refresh_token'),
  set: (a: string, r: string) => {
    localStorage.setItem('access_token', a)
    localStorage.setItem('refresh_token', r)
  },
  clear: () => {
    localStorage.removeItem('access_token')
    localStorage.removeItem('refresh_token')
  },
}

// ── Types ─────────────────────────────────────────────────────

export interface TokenResponse { access_token: string; refresh_token: string; token_type: string }
export interface User { id: string; email: string; full_name: string; role: 'user' | 'admin' }

export interface SimilarCustomer {
  similarity_score: number
  churn_label: number
  distance: number
}

export interface PredictionResponse {
  prediction_id: string
  customer_id: string | null
  churn_probability: number
  risk_level: 'low' | 'medium' | 'high'
  model_version: string
  inference_time_ms: number
  similar_customers: SimilarCustomer[]
  recommendation: string
  created_at: string
}

export interface BatchSummary {
  job_id: string
  filename: string
  total_rows: number
  high_risk: number
  medium_risk: number
  low_risk: number
  avg_churn_probability: number
  status: string
  created_at: string
}

export interface ModelInfo {
  version: string
  auc_roc: number
  precision: number
  recall: number
  f1_score: number
  training_rows: number
  churn_rate: number
  feature_count: number
}

export interface Analytics {
  total_predictions: number
  high_risk_count: number
  medium_risk_count: number
  low_risk_count: number
  avg_churn_probability: number
  avg_inference_time_ms: number
  predictions_today: number
  predictions_this_week: number
  daily_volume: { date: string; predictions: number }[]
  risk_distribution: { risk: string; count: number }[]
  churn_probability_histogram: { range: string; count: number }[]
  top_batch_jobs: any[]
}

export interface DriftReport {
  drift_detected: boolean
  max_psi: number
  avg_psi: number
  drifted_features: Record<string, number>
  monitored_features: Record<string, number>
  stable_feature_count: number
  threshold: number
  n_reference: number
  n_current: number
  checked_at: string
}

// ── Core fetch ────────────────────────────────────────────────

async function apiFetch<T>(path: string, options: RequestInit = {}, retried = false): Promise<T> {
  const token = tokenStorage.getAccess()
  const isForm = options.body instanceof FormData
  const headers: Record<string, string> = {
    ...(isForm ? {} : { 'Content-Type': 'application/json' }),
    ...(token  ? { Authorization: `Bearer ${token}` } : {}),
    ...(options.headers as Record<string, string> | undefined),
  }
  const res = await fetch(`${API}${path}`, { ...options, headers })

  if (res.status === 401 && !retried) {
    const ok = await tryRefresh()
    if (ok) return apiFetch<T>(path, options, true)
    tokenStorage.clear()
    window.location.href = '/login'
    throw new Error('Session expired')
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: 'Request failed' }))
    throw new Error(body.detail ?? 'Request failed')
  }
  if (res.status === 204) return undefined as T
  return res.json()
}

async function tryRefresh(): Promise<boolean> {
  const r = tokenStorage.getRefresh()
  if (!r) return false
  try {
    const res = await fetch(`${API}/auth/refresh`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ refresh_token: r }),
    })
    if (!res.ok) return false
    const d: TokenResponse = await res.json()
    tokenStorage.set(d.access_token, d.refresh_token)
    return true
  } catch { return false }
}

// ── Auth ──────────────────────────────────────────────────────

export const authApi = {
  register: (email: string, full_name: string, password: string) =>
    apiFetch<TokenResponse>('/auth/register', { method: 'POST', body: JSON.stringify({ email, full_name, password }) }),
  login: (email: string, password: string) =>
    apiFetch<TokenResponse>('/auth/login', { method: 'POST', body: JSON.stringify({ email, password }) }),
  logout: () => {
    const r = tokenStorage.getRefresh()
    if (r) fetch(`${API}/auth/logout`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ refresh_token: r }) }).catch(() => {})
    tokenStorage.clear()
  },
  me: () => apiFetch<User>('/auth/me'),
}

// ── Predict ───────────────────────────────────────────────────

export const predictApi = {
  single: (payload: object) =>
    apiFetch<PredictionResponse>('/predict/single', { method: 'POST', body: JSON.stringify(payload) }),
  batch: (file: File) => {
    const form = new FormData()
    form.append('file', file)
    return apiFetch<BatchSummary>('/predict/batch', { method: 'POST', body: form })
  },
}

// ── Admin ─────────────────────────────────────────────────────

export const adminApi = {
  getModel:     () => apiFetch<ModelInfo>('/admin/model'),
  getAnalytics: () => apiFetch<Analytics>('/admin/analytics'),
  getDrift:     () => apiFetch<DriftReport>('/admin/drift'),
  getHistory:   (limit = 100, offset = 0) =>
    apiFetch<{ total: number; items: any[] }>(`/admin/predictions/history?limit=${limit}&offset=${offset}`),
}
