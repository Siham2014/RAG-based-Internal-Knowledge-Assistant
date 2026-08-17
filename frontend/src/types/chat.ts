export type Language = 'en' | 'fr' | 'ar'

export interface Source { rank: number; source: string; chunk_id: string; page_number: number | null; source_url: string | null; reranker_score: number; rrf_score: number }
export interface Timings { retrieval_time_ms: number; confidence_time_ms: number; generation_time_ms: number; citation_validation_time_ms: number; total_time_ms: number }
export interface AskResponse { accepted: boolean; answer: string; provider: string | null; model_name: string | null; confidence_score: number; failed_rules: string[]; citations: string[]; sources: Source[]; timings: Timings; refusal_reason: string | null }
export interface Message { id: string; role: 'user' | 'assistant'; content: string; createdAt: string; metadata?: AskResponse; error?: boolean }
export interface Conversation { id: string; title: string; messages: Message[]; createdAt: string; favorite?: boolean }
