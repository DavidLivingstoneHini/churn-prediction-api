import React, { createContext, useCallback, useContext, useEffect, useState } from 'react'
import { authApi, tokenStorage, type User } from '../api/client'

interface AuthCtx {
  user: User | null
  loading: boolean
  login: (email: string, password: string) => Promise<void>
  register: (email: string, fullName: string, password: string) => Promise<void>
  logout: () => void
  isAdmin: boolean
}

const AuthContext = createContext<AuthCtx | null>(null)

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser]       = useState<User | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (!tokenStorage.getAccess()) { setLoading(false); return }
    authApi.me().then(setUser).catch(() => tokenStorage.clear()).finally(() => setLoading(false))
  }, [])

  const login = useCallback(async (email: string, password: string) => {
    const t = await authApi.login(email, password)
    tokenStorage.set(t.access_token, t.refresh_token)
    setUser(await authApi.me())
  }, [])

  const register = useCallback(async (email: string, fullName: string, password: string) => {
    const t = await authApi.register(email, fullName, password)
    tokenStorage.set(t.access_token, t.refresh_token)
    setUser(await authApi.me())
  }, [])

  const logout = useCallback(() => { authApi.logout(); setUser(null) }, [])

  return (
    <AuthContext.Provider value={{ user, loading, login, register, logout, isAdmin: user?.role === 'admin' }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used within AuthProvider')
  return ctx
}
