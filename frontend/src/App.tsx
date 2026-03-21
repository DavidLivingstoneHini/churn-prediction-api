import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { AuthProvider, useAuth } from './context/AuthContext'
import LoginPage from './pages/LoginPage'
import RegisterPage from './pages/RegisterPage'
import PredictPage from './pages/PredictPage'
import BatchPage from './pages/BatchPage'
import DashboardLayout from './pages/dashboard/DashboardLayout'
import AnalyticsPage from './pages/dashboard/AnalyticsPage'
import DriftPage from './pages/dashboard/DriftPage'
import HistoryPage from './pages/dashboard/HistoryPage'

function Spinner() {
  return (
    <div className="flex items-center justify-center min-h-screen">
      <div className="w-8 h-8 border-2 border-brand-600 border-t-transparent rounded-full animate-spin" />
    </div>
  )
}

function Protected({ children }: { children: React.ReactNode }) {
  const { user, loading } = useAuth()
  if (loading) return <Spinner />
  if (!user) return <Navigate to="/login" replace />
  return <>{children}</>
}

function AdminOnly({ children }: { children: React.ReactNode }) {
  const { user, loading, isAdmin } = useAuth()
  if (loading) return <Spinner />
  if (!user) return <Navigate to="/login" replace />
  if (!isAdmin) return <Navigate to="/predict" replace />
  return <>{children}</>
}

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/login"    element={<LoginPage />} />
          <Route path="/register" element={<RegisterPage />} />
          <Route path="/predict"  element={<Protected><PredictPage /></Protected>} />
          <Route path="/batch"    element={<Protected><BatchPage /></Protected>} />
          <Route path="/dashboard" element={<AdminOnly><DashboardLayout /></AdminOnly>}>
            <Route index element={<Navigate to="analytics" replace />} />
            <Route path="analytics" element={<AnalyticsPage />} />
            <Route path="drift"     element={<DriftPage />} />
            <Route path="history"   element={<HistoryPage />} />
          </Route>
          <Route path="*" element={<Navigate to="/predict" replace />} />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  )
}
