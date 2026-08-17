# Nexus AI Frontend

Dark-mode React client for the existing FastAPI knowledge assistant.

## Run locally

```powershell
npm.cmd install
npm.cmd run dev
```

Open `http://127.0.0.1:5500`. This port is intentional: it matches the origins currently allowed by the frozen FastAPI CORS configuration. The API base URL defaults to `http://127.0.0.1:8001` and can be overridden with `VITE_API_BASE_URL`.

## Production build

```powershell
npm.cmd run build
```

Document upload, Arabic responses, and writable settings are presented as **Coming Soon** because the current backend does not expose compatible functionality. Conversation history is stored locally in the browser.
