import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import { Sparkles, Mail, Lock, User, ArrowRight, AlertCircle, Eye, EyeOff } from 'lucide-react'

export default function Login() {
  const [mode, setMode]           = useState('signin')
  const [email, setEmail]         = useState('')
  const [password, setPassword]   = useState('')
  const [fullName, setFullName]   = useState('')
  const [showPass, setShowPass]   = useState(false)
  const [error, setError]         = useState(null)
  const [loading, setLoading]     = useState(false)
  const [success, setSuccess]     = useState(false)

  const { signIn, signUp } = useAuth()
  const navigate = useNavigate()

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError(null)
    setLoading(true)
    try {
      if (mode === 'signin') {
        const { error } = await signIn(email, password)
        if (error) throw error
        navigate('/')
      } else {
        if (!fullName.trim()) throw new Error('Full name is required')
        if (password.length < 6) throw new Error('Password must be at least 6 characters')
        const { error } = await signUp(email, password, fullName)
        if (error) throw error
        setSuccess(true)
      }
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div
      style={{
        minHeight: '100vh',
        background: 'var(--bg-base)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '24px',
        fontFamily: 'var(--font-body)',
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      {/* Background orbs */}
      <div className="bg-orbs" />
      <div className="bg-orb-cyan" />

      {/* Grid pattern */}
      <div
        style={{
          position: 'fixed',
          inset: 0,
          backgroundImage: `
            linear-gradient(rgba(139,92,246,0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(139,92,246,0.03) 1px, transparent 1px)
          `,
          backgroundSize: '48px 48px',
          pointerEvents: 'none',
          zIndex: 0,
        }}
      />

      <div style={{ width: '100%', maxWidth: '440px', position: 'relative', zIndex: 10 }}>
        {/* Logo */}
        <div style={{ textAlign: 'center', marginBottom: '36px' }}>
          <div
            style={{
              width: '60px', height: '60px',
              background: 'linear-gradient(135deg, #F59E0B, #D97706)',
              borderRadius: '18px',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              margin: '0 auto 18px',
              boxShadow: '0 0 40px rgba(245,158,11,0.4), 0 0 80px rgba(245,158,11,0.15)',
              position: 'relative',
            }}
          >
            <Sparkles size={28} color="#06060E" />
            <div
              style={{
                position: 'absolute',
                inset: '-6px',
                borderRadius: '24px',
                border: '1px solid rgba(245,158,11,0.2)',
                animation: 'glow-pulse 3s ease-in-out infinite',
              }}
            />
          </div>
          <h1
            style={{
              fontFamily: 'var(--font-display)',
              fontSize: '28px',
              fontWeight: 800,
              letterSpacing: '-0.5px',
              marginBottom: '6px',
              background: 'linear-gradient(135deg, #EEEEFF, #8B5CF6 50%, #F59E0B)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}
          >
            Research AI
          </h1>
          <p style={{ fontSize: '13px', color: 'var(--text-muted)' }}>
            {mode === 'signin' ? 'Sign in to your research workspace' : 'Create your research workspace'}
          </p>
        </div>

        {/* Card */}
        <div
          className="glass"
          style={{
            padding: '32px',
            border: '1px solid var(--border-subtle)',
            position: 'relative',
            overflow: 'hidden',
          }}
        >
          {/* Card top glow line */}
          <div
            style={{
              position: 'absolute',
              top: 0, left: 0, right: 0, height: '1px',
              background: 'linear-gradient(90deg, transparent, rgba(139,92,246,0.6), rgba(245,158,11,0.4), transparent)',
            }}
          />

          {/* Mode tabs */}
          <div
            style={{
              display: 'flex',
              background: 'rgba(255,255,255,0.03)',
              borderRadius: '12px',
              padding: '4px',
              marginBottom: '28px',
              border: '1px solid var(--border-subtle)',
              gap: '4px',
            }}
          >
            {[
              { id: 'signin', label: 'Sign In' },
              { id: 'signup', label: 'Create Account' },
            ].map((m) => (
              <button
                key={m.id}
                onClick={() => { setMode(m.id); setError(null); setSuccess(false) }}
                style={{
                  flex: 1,
                  padding: '9px',
                  borderRadius: '9px',
                  border: 'none',
                  cursor: 'pointer',
                  fontSize: '13px',
                  fontFamily: 'var(--font-display)',
                  fontWeight: 600,
                  transition: 'all 0.25s ease',
                  background: mode === m.id
                    ? 'linear-gradient(135deg, rgba(139,92,246,0.2), rgba(245,158,11,0.1))'
                    : 'transparent',
                  color: mode === m.id ? 'var(--text-primary)' : 'var(--text-muted)',
                  boxShadow: mode === m.id ? '0 2px 8px rgba(0,0,0,0.3)' : 'none',
                  letterSpacing: '0.01em',
                }}
              >
                {m.label}
              </button>
            ))}
          </div>

          {/* Success */}
          {success && (
            <div
              style={{
                background: 'var(--emerald-dim)',
                border: '1px solid var(--emerald-border)',
                borderRadius: '10px',
                padding: '12px 14px',
                marginBottom: '18px',
                fontSize: '13px',
                color: 'var(--emerald)',
                boxShadow: '0 0 16px rgba(16,185,129,0.1)',
              }}
            >
              ✓ Account created! You can now sign in.
            </div>
          )}

          {/* Error */}
          {error && (
            <div
              style={{
                background: 'var(--rose-dim)',
                border: '1px solid var(--rose-border)',
                borderRadius: '10px',
                padding: '12px 14px',
                marginBottom: '18px',
                fontSize: '13px',
                color: 'var(--rose)',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                boxShadow: '0 0 16px rgba(244,63,94,0.1)',
              }}
            >
              <AlertCircle size={14} />
              {error}
            </div>
          )}

          {/* Form */}
          <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            {mode === 'signup' && (
              <div>
                <label style={{ fontSize: '11px', color: 'var(--text-muted)', display: 'block', marginBottom: '7px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.8px', fontFamily: 'var(--font-display)' }}>
                  Full Name
                </label>
                <div style={{ position: 'relative' }}>
                  <User size={15} style={{ position: 'absolute', left: '13px', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)', zIndex: 1 }} />
                  <input
                    type="text"
                    value={fullName}
                    onChange={e => setFullName(e.target.value)}
                    placeholder="Dr. Jane Smith"
                    required
                    className="input-field"
                    style={{ paddingLeft: '38px' }}
                  />
                </div>
              </div>
            )}

            <div>
              <label style={{ fontSize: '11px', color: 'var(--text-muted)', display: 'block', marginBottom: '7px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.8px', fontFamily: 'var(--font-display)' }}>
                Email Address
              </label>
              <div style={{ position: 'relative' }}>
                <Mail size={15} style={{ position: 'absolute', left: '13px', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)', zIndex: 1 }} />
                <input
                  type="email"
                  value={email}
                  onChange={e => setEmail(e.target.value)}
                  placeholder="you@university.edu"
                  required
                  className="input-field"
                  style={{ paddingLeft: '38px' }}
                />
              </div>
            </div>

            <div>
              <label style={{ fontSize: '11px', color: 'var(--text-muted)', display: 'block', marginBottom: '7px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.8px', fontFamily: 'var(--font-display)' }}>
                Password
              </label>
              <div style={{ position: 'relative' }}>
                <Lock size={15} style={{ position: 'absolute', left: '13px', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)', zIndex: 1 }} />
                <input
                  type={showPass ? 'text' : 'password'}
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  placeholder={mode === 'signup' ? 'Min. 6 characters' : '••••••••'}
                  required
                  className="input-field"
                  style={{ paddingLeft: '38px', paddingRight: '42px' }}
                />
                <button
                  type="button"
                  onClick={() => setShowPass(!showPass)}
                  style={{
                    position: 'absolute', right: '12px', top: '50%',
                    transform: 'translateY(-50%)',
                    background: 'none', border: 'none', cursor: 'pointer',
                    color: 'var(--text-muted)', padding: '2px',
                  }}
                >
                  {showPass ? <EyeOff size={14} /> : <Eye size={14} />}
                </button>
              </div>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="btn-amber"
              style={{
                width: '100%',
                justifyContent: 'center',
                padding: '12px 24px',
                fontSize: '14px',
                marginTop: '6px',
                borderRadius: '12px',
                letterSpacing: '0.02em',
              }}
            >
              {loading ? (
                <><div className="spinner w-4 h-4" style={{ borderTopColor: '#06060E', borderColor: 'rgba(0,0,0,0.2)' }} /> Processing…</>
              ) : (
                <>
                  {mode === 'signin' ? 'Sign In to Workspace' : 'Create Account'}
                  <ArrowRight size={15} />
                </>
              )}
            </button>
          </form>
        </div>

        <p style={{ textAlign: 'center', marginTop: '24px', fontSize: '11px', color: 'var(--text-faint)', fontFamily: 'var(--font-mono)' }}>
          Research AI · Literature Intelligence Platform
        </p>
      </div>
    </div>
  )
}
