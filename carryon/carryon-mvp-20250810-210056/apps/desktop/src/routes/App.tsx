import React, { useState } from 'react'
import { PrimerPreview } from './PrimerPreview'
import { Dashboard } from './Dashboard'

export function App() {
  const [route, setRoute] = useState<'dashboard' | 'primer'>('dashboard')
  return (
    <div style={{ fontFamily: 'system-ui, sans-serif', padding: 16 }}>
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h1>CarryOn</h1>
        <nav>
          <button onClick={() => setRoute('dashboard')}>Dashboard</button>
          <button onClick={() => setRoute('primer')}>Primer</button>
        </nav>
      </header>
      <main>
        {route === 'dashboard' ? <Dashboard /> : <PrimerPreview />}
      </main>
    </div>
  )
}
