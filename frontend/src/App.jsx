import { useEffect, useRef, useState } from 'react'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function getOrCreateSessionId() {
  let id = localStorage.getItem('college_chatbot_session_id')
  if (!id) {
    id = crypto.randomUUID()
    localStorage.setItem('college_chatbot_session_id', id)
  }
  return id
}

export default function App() {
  const [about, setAbout] = useState(null)
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [sessionId, setSessionId] = useState(getOrCreateSessionId)
  const logRef = useRef(null)

  useEffect(() => {
    fetch(`${API_URL}/api/about`)
      .then((r) => r.json())
      .then(setAbout)
      .catch(() => setError('Could not reach the backend. Is it running on ' + API_URL + '?'))
  }, [])

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight, behavior: 'smooth' })
  }, [messages, loading])

  async function sendMessage(e) {
    e.preventDefault()
    const text = input.trim()
    if (!text || loading) return

    setMessages((m) => [...m, { role: 'user', content: text }])
    setInput('')
    setLoading(true)
    setError('')

    try {
      const res = await fetch(`${API_URL}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, session_id: sessionId }),
      })
      if (!res.ok) throw new Error(`Server responded ${res.status}`)
      const data = await res.json()
      setSessionId(data.session_id)
      setMessages((m) => [...m, { role: 'assistant', content: data.response }])
    } catch (err) {
      setError(err.message || 'Something went wrong reaching the chatbot.')
      setMessages((m) => [
        ...m,
        { role: 'assistant', content: 'Sorry — I could not reach the server. Please try again.', isError: true },
      ])
    } finally {
      setLoading(false)
    }
  }

  function newConversation() {
    fetch(`${API_URL}/api/session/reset?session_id=${sessionId}`, { method: 'POST' }).catch(() => { })
    const id = crypto.randomUUID()
    localStorage.setItem('college_chatbot_session_id', id)
    setSessionId(id)
    setMessages([])
    setError('')
  }

  const userTurns = messages.filter((m) => m.role === 'user')

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="sidebar-brand">
          <img src="/logo.jpg" alt="Logo" className="mark" />
          <h1>{about?.title || 'IIT Guwahati Chatbot'}</h1>
        </div>

        <div className="card">
          <p className="card-label">Description</p>
          <p>{about?.description || 'Loading…'}</p>
        </div>

        <div className="card">
          <p className="card-label">Goals</p>
          <ul>
            {(about?.goals || []).map((g) => (
              <li key={g}>{g}</li>
            ))}
          </ul>
        </div>

        <div className="card">
          <p className="card-label">Purpose</p>
          <p>{about?.purpose}</p>
        </div>

        <div className="card">
          <p className="card-label">Our Values</p>
          <ul>
            {(about?.values || []).map((v) => (
              <li key={v}>{v}</li>
            ))}
          </ul>
        </div>

        <div className="card">
          <p className="card-label">This Conversation</p>
          {userTurns.length === 0 && <p style={{ color: 'var(--muted)' }}>No questions asked yet.</p>}
          {userTurns.map((m, i) => (
            <div key={i} className="history-item">
              {i + 1}. {m.content.slice(0, 40)}
              {m.content.length > 40 ? '…' : ''}
            </div>
          ))}
        </div>

        <div className="sidebar-actions">
          <button className="btn-secondary" onClick={newConversation} disabled={messages.length === 0}>
            Start new conversation
          </button>
        </div>
      </aside>

      <main className="chat-column">
        <header className="chat-header">
          <h2>Ask about {about?.title ? about.title.replace(' Chatbot', '') : 'your college'}</h2>
          <span className="subtitle">session · {sessionId.slice(0, 8)}</span>
        </header>

        {error && <div className="status-banner">{error}</div>}

        <div className="chat-log" ref={logRef}>
          {messages.length === 0 && !loading && (
            <div className="empty-state">
              <img src="/logo.jpg" alt="Logo" className="mark-lg" />
              <h3>Ask a question about IIT Guwahati</h3>
              <p>Answers are grounded in the college's own documents - admissions, programs, campus services, and more.</p>
            </div>
          )}

          {messages.map((m, i) => (
            <div key={i} className={`msg-row ${m.role}`}>
              <div className={`avatar ${m.role === 'user' ? 'user' : 'bot'}`}>
                {m.role === 'user' ? 'YOU' : <img src="/logo.jpg" alt="AI" className="avatar-logo" />}
              </div>
              <div className={`bubble ${m.isError ? 'error' : ''}`}>{m.content}</div>
            </div>
          ))}

          {loading && (
            <div className="msg-row assistant">
              <div className="avatar bot"><img src="/logo.jpg" alt="AI" className="avatar-logo" /></div>
              <div className="bubble">
                <div className="typing">
                  <span></span><span></span><span></span>
                </div>
              </div>
            </div>
          )}
        </div>

        <form className="chat-input-bar" onSubmit={sendMessage}>
          <div className="chat-input-row">
            <input
              type="text"
              placeholder="Ask a question about IIT Guwahati"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={loading}
              autoFocus
            />
            <button type="submit" disabled={loading || !input.trim()}>
              Send
            </button>
          </div>
        </form>
      </main>
    </div>
  )
}