import React, { useState } from 'react'

export function PrimerPreview() {
  const [query, setQuery] = useState('Say hello like Diane would')
  const [resp, setResp] = useState<any>(null)
  async function build() {
    const r = await fetch('http://localhost:8000/v1/prime', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, k: 8 })
    })
    const j = await r.json()
    setResp(j)
  }
  return (
    <section>
      <h2>Session Primer</h2>
      <input value={query} onChange={e=>setQuery(e.target.value)} style={{ width: '60%' }} />
      <button onClick={build}>Build</button>
      <pre style={{ background: '#111', color: '#0f0', padding: 12, marginTop: 12 }}>
        {resp ? JSON.stringify(resp, null, 2) : 'No primer yet.'}
      </pre>
    </section>
  )
}
