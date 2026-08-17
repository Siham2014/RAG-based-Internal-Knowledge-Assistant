import axios from 'axios'
import type { AskResponse, Language } from '../types/chat'

const api = axios.create({ baseURL: import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8001', timeout: 120_000, headers: { 'Content-Type': 'application/json' } })

export async function askAssistant(question: string, language: Language) {
  const { data } = await api.post<AskResponse>('/ask', { question, language })
  return data
}

export function readableApiError(error: unknown) {
  if (!axios.isAxiosError(error)) return 'Something went wrong. Please try again.'
  if (!error.response) return 'Unable to reach the assistant. Make sure the API is running on port 8001.'
  const detail = error.response.data?.detail
  if (typeof detail === 'string') return detail
  if (typeof detail?.message === 'string') return detail.message
  return `Request failed (${error.response.status}). Please try again.`
}
