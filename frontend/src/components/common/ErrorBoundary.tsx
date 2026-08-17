import { Component, type ErrorInfo, type ReactNode } from 'react'
import { FiAlertTriangle, FiRefreshCw } from 'react-icons/fi'

interface Props {
  children: ReactNode
}

interface State {
  hasError: boolean
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false }

  static getDerivedStateFromError(): State {
    return { hasError: true }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('Nexus AI interface error', error, info)
  }

  render() {
    if (this.state.hasError) {
      return (
        <main className="error-fallback">
          <div className="error-fallback-card">
            <span><FiAlertTriangle /></span>
            <h1>Something went wrong</h1>
            <p>The interface encountered an unexpected error. Your conversations remain stored in this browser.</p>
            <button onClick={() => window.location.reload()}>
              <FiRefreshCw /> Reload application
            </button>
          </div>
        </main>
      )
    }

    return this.props.children
  }
}
