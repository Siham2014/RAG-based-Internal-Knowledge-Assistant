async function relayAsk(request) {
  const origin = process.env.RAG_BACKEND_URL;
  const token = process.env.RAG_BACKEND_TOKEN;
  if (!origin || !token) {
    return Response.json({ detail: 'The RAG backend is not configured for this deployment.' }, { status: 503 });
  }

  let endpoint;
  try {
    endpoint = new URL('/ask', origin);
    if (endpoint.protocol !== 'https:' || endpoint.username || endpoint.password) {
      throw new Error('Invalid backend URL');
    }
  } catch {
    return Response.json({ detail: 'The RAG backend URL is invalid.' }, { status: 503 });
  }

  if (request.headers.get('content-type')?.split(';', 1)[0]?.trim() !== 'application/json') {
    return Response.json({ detail: 'Expected application/json.' }, { status: 415 });
  }

  try {
    const upstream = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
      body: await request.text(),
      signal: AbortSignal.timeout(110_000),
      redirect: 'error',
    });
    if (upstream.status >= 500 || upstream.status === 401 || upstream.status === 403 || !upstream.headers.get('content-type')?.startsWith('application/json')) {
      throw new Error('Backend unavailable');
    }
    return new Response(upstream.body, {
      status: upstream.status,
      headers: { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' },
    });
  } catch {
    return Response.json({ detail: 'The RAG backend is unavailable. Please try again later.' }, { status: 502 });
  }
}

export default {
  fetch(request) {
    if (request.method === 'POST') return relayAsk(request);
    return Response.json({ detail: 'Method not allowed.' }, { status: 405, headers: { Allow: 'POST' } });
  },
};
