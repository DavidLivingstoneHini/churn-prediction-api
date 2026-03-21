import { NavLink, Outlet, useNavigate } from 'react-router-dom'
import clsx from 'clsx'
import { Activity, BarChart2, Clock, GitBranch, LogOut, Upload, User } from 'lucide-react'
import { useAuth } from '../../context/AuthContext'

const nav = [
  { to: '/dashboard/analytics', icon: BarChart2, label: 'Analytics' },
  { to: '/dashboard/drift',     icon: GitBranch,  label: 'Drift Monitor' },
  { to: '/dashboard/history',   icon: Clock,      label: 'Audit Log' },
]

export default function DashboardLayout() {
  const { user, logout } = useAuth()
  const navigate = useNavigate()

  return (
    <div className="flex h-screen bg-gray-50">
      <aside className="w-56 bg-gray-900 flex flex-col flex-shrink-0">
        <div className="flex items-center gap-2 px-4 py-4 border-b border-gray-800">
          <div className="w-7 h-7 bg-brand-600 rounded-lg flex items-center justify-center">
            <Activity className="w-4 h-4 text-white" />
          </div>
          <div>
            <p className="text-white text-sm font-semibold">Churn API</p>
            <p className="text-gray-500 text-xs">Admin Dashboard</p>
          </div>
        </div>

        <nav className="flex-1 px-3 py-4 space-y-1">
          {nav.map(({ to, icon: Icon, label }) => (
            <NavLink key={to} to={to}
              className={({ isActive }) => clsx(
                'flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-colors',
                isActive
                  ? 'bg-gray-700 text-white'
                  : 'text-gray-400 hover:bg-gray-800 hover:text-gray-200'
              )}
            >
              <Icon className="w-4 h-4 flex-shrink-0" />{label}
            </NavLink>
          ))}

          <div className="pt-2 border-t border-gray-800 mt-2 space-y-1">
            <button onClick={() => navigate('/predict')}
              className="w-full flex items-center gap-2 px-3 py-2 rounded-lg text-sm text-gray-400 hover:bg-gray-800 hover:text-gray-200 transition-colors">
              <User className="w-4 h-4 flex-shrink-0" />Single Predict
            </button>
            <button onClick={() => navigate('/batch')}
              className="w-full flex items-center gap-2 px-3 py-2 rounded-lg text-sm text-gray-400 hover:bg-gray-800 hover:text-gray-200 transition-colors">
              <Upload className="w-4 h-4 flex-shrink-0" />Batch Predict
            </button>
          </div>
        </nav>

        <div className="border-t border-gray-800 p-3">
          <div className="flex items-center gap-2 px-3 py-2">
            <div className="w-6 h-6 bg-brand-600 rounded-full flex items-center justify-center flex-shrink-0">
              <span className="text-white text-xs font-semibold">
                {user?.full_name?.[0]?.toUpperCase() ?? '?'}
              </span>
            </div>
            <span className="text-xs text-gray-400 truncate flex-1">{user?.full_name}</span>
            <button onClick={() => { logout(); navigate('/login') }}
              className="text-gray-600 hover:text-gray-300 transition-colors flex-shrink-0">
              <LogOut className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </aside>

      <main className="flex-1 overflow-auto">
        <Outlet />
      </main>
    </div>
  )
}
