import { Component } from 'react';

// Last line of defence: a render-time crash anywhere in the tree shows this
// message instead of a blank page. Plain inline styles on purpose — if the
// stylesheet itself failed to load, the message must still be readable.
export default class ErrorBoundary extends Component {
  state = { error: null };

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    console.error('[ErrorBoundary] render crash', error, info?.componentStack);
  }

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;

    return (
      <div
        role="alert"
        data-testid="fatal-error"
        style={{
          minHeight: '100vh',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '24px',
          background: '#05060a',
          color: '#e5e7eb',
          fontFamily: 'system-ui, sans-serif',
        }}
      >
        <div style={{ maxWidth: 480, textAlign: 'center' }}>
          <h1 style={{ fontSize: 20, marginBottom: 8 }}>Something went wrong</h1>
          <p style={{ color: '#9ca3af', marginBottom: 16 }}>
            Plastic Decoder hit an unexpected error and had to stop. Your images were not uploaded anywhere.
          </p>
          <code style={{ display: 'block', color: '#fbbf24', marginBottom: 16, wordBreak: 'break-word' }}>
            {String(error?.message ?? error)}
          </code>
          <button
            type="button"
            onClick={() => window.location.reload()}
            style={{ padding: '8px 16px', borderRadius: 8, border: '1px solid #34d399', background: 'transparent', color: '#6ee7b7', cursor: 'pointer' }}
          >
            Reload
          </button>
        </div>
      </div>
    );
  }
}
