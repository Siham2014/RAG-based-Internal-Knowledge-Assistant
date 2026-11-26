# Vercel deployment

Use `frontend` as the root directory, with the Vite preset and Node.js 22 or later. The production build sends chat requests to `/api/ask`.

For a working chat, deploy the existing FastAPI app in a persistent container. Provision PostgreSQL with pgvector and index the corpus using `scripts/ingest_pgvector.py`. Keep the committed BM25 index and make the embedding and reranking models available. Configure the LLM credentials used by the backend without exposing them to the frontend. Put an HTTPS gateway in front of FastAPI that verifies a bearer token before forwarding requests. Keep the Vercel deployment protected when it uses private documents or paid inference.

Set server-side Vercel variables `RAG_BACKEND_URL` (HTTPS gateway origin) and `RAG_BACKEND_TOKEN` (gateway bearer token) in the chosen environment. Redeploy after configuring them. The relay returns 503 until both variables are set.

Check `POST /api/ask` with a known in-corpus question. Verify that the response has `accepted: true`, a grounded answer, and sources. Submit an out-of-corpus question to verify the confidence gate, then send a follow-up with the returned conversation ID.
