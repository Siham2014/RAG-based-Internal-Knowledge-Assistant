import { useEffect, useMemo, useState } from 'react'
import { askAssistant, readableApiError } from '../services/api'
import type { Conversation, Language, Message } from '../types/chat'

const STORAGE_KEY = 'nexus-ai-conversations'
const createConversation = (): Conversation => ({ id: crypto.randomUUID(), title: 'New conversation', messages: [], createdAt: new Date().toISOString() })

export function useChat() {
  const [conversations, setConversations] = useState<Conversation[]>(() => { try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]') } catch { return [] } })
  const [activeId, setActiveId] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const active = useMemo(() => conversations.find((item) => item.id === activeId), [conversations, activeId])

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations))
  }, [conversations])

  const newChat = () => setActiveId(null)
  const patch = (id: string, fn: (chat: Conversation) => Conversation) => setConversations((items) => items.map((item) => item.id === id ? fn(item) : item))

  const send = (text: string, language: Language, conversationId?: string) => {
    const content = text.trim(); if (!content || isLoading) return undefined
    const id = conversationId ?? crypto.randomUUID(); const now = new Date().toISOString()
    const userMessage: Message = { id: crypto.randomUUID(), role: 'user', content, createdAt: now }
    if (conversationId) {
      patch(id, (chat) => ({ ...chat, messages: [...chat.messages, userMessage] }))
    } else {
      const conversation: Conversation = { ...createConversation(), id, title: content.slice(0, 42), messages: [userMessage], createdAt: now }
      setConversations((current) => [conversation, ...current])
    }
    setActiveId(id)
    setIsLoading(true)
    const request = async () => {
      try {
        const result = await askAssistant(content, language)
        const assistant: Message = { id: crypto.randomUUID(), role: 'assistant', content: result.answer || result.refusal_reason || 'No response was generated.', createdAt: new Date().toISOString(), metadata: result }
        patch(id, (chat) => ({ ...chat, messages: [...chat.messages, assistant] }))
      } catch (error) {
        const assistant: Message = { id: crypto.randomUUID(), role: 'assistant', content: readableApiError(error), createdAt: new Date().toISOString(), error: true }
        patch(id, (chat) => ({ ...chat, messages: [...chat.messages, assistant] }))
      } finally { setIsLoading(false) }
    }
    void request()
    return id
  }

  return { conversations, active, activeId, setActiveId, isLoading, newChat, send }
}
