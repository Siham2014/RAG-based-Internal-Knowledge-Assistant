import { useState } from 'react'
import { Navigate, Route, Routes, useNavigate, useParams } from 'react-router-dom'
import { AnimatePresence, motion } from 'framer-motion'
import { ArcAnimation } from './components/hero/ArcAnimation'
import { RobotHero } from './components/hero/RobotHero'
import { Sidebar } from './components/layout/Sidebar'
import { Header } from './components/layout/Header'
import { ChatWindow } from './components/chat/ChatWindow'
import { MessageInput } from './components/chat/MessageInput'
import { NewChatButton } from './components/chat/NewChatButton'
import { InfoDrawer } from './components/drawers/InfoDrawer'
import { SettingsDrawer } from './components/drawers/SettingsDrawer'
import { UploadModal } from './components/modals/UploadModal'
import { useChat } from './hooks/useChat'
import { FiBookOpen, FiMessageSquare, FiStar } from 'react-icons/fi'
import type { AskResponse, Conversation, Language, Message } from './types/chat'

function AssistantPage({ conversation, loading, language, onSend, onUpload, onInfo, onNew }: { conversation?: Conversation; loading: boolean; language: Language; onSend: (value: string, language: Language) => void; onUpload: () => void; onInfo: (message: Message) => void; onNew: () => void }) {
  const hasMessages = !!conversation?.messages.length
  const suggestions = ['How can I improve Azure reliability?', 'Explain the Well-Architected Framework', 'What are the cloud security best practices?']
  return <div className={`content-scroll ${hasMessages ? 'chat-mode' : ''}`}><AnimatePresence mode="wait">{!hasMessages ? <motion.section className="empty-state" key="hero" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0, y: -20 }}><ArcAnimation /><RobotHero /><div className="hero-copy"><span className="eyebrow"><i /> KNOWLEDGE, AMPLIFIED</span><h1>How can I help you<span>?</span></h1><p>Ask questions. Find clarity. Move faster.</p></div><div className="hero-composer"><MessageInput onSend={(value) => onSend(value, language)} onUpload={onUpload} disabled={loading} /><div className="question-suggestions">{suggestions.map((suggestion) => <button key={suggestion} disabled={loading} onClick={() => onSend(suggestion, language)}>{suggestion}</button>)}</div><small>Nexus AI can make mistakes. Verify important information.</small></div></motion.section> : <motion.section className="conversation" key="chat" initial={{ opacity: 0 }} animate={{ opacity: 1 }}><ChatWindow messages={conversation?.messages ?? []} loading={loading} onInfo={onInfo} /><div className="sticky-composer"><MessageInput onSend={(value) => onSend(value, language)} onUpload={onUpload} disabled={loading} /><small>Nexus AI can make mistakes. Verify important information.</small></div></motion.section>}</AnimatePresence><NewChatButton onClick={onNew} /></div>
}

function WorkspacePage({ icon, eyebrow, title, description, children }: { icon: React.ReactNode; eyebrow: string; title: string; description: string; children: React.ReactNode }) {
  return <div className="content-scroll"><motion.section className="workspace-page" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}><div className="workspace-heading"><span>{icon}</span><div><small>{eyebrow}</small><h1>{title}</h1><p>{description}</p></div></div>{children}</motion.section></div>
}

function EmptyCollection({ text, badge }: { text: string; badge?: string }) {
  return <div className="empty-collection">{badge && <span>{badge}</span>}<p>{text}</p></div>
}

function ConversationsPage({ conversations, activeId, onSelect }: { conversations: Conversation[]; activeId: string | null; onSelect: (id: string) => void }) {
  return <WorkspacePage icon={<FiMessageSquare />} eyebrow="Your workspace" title="Conversations" description="Continue where you left off.">{conversations.length ? <div className="conversation-list">{conversations.map((conversation) => <button key={conversation.id} className={conversation.id === activeId ? 'active' : ''} onClick={() => onSelect(conversation.id)}><span><FiMessageSquare /></span><div><strong>{conversation.title}</strong><small>{conversation.messages.length} messages</small></div><time>{new Date(conversation.createdAt).toLocaleDateString()}</time></button>)}</div> : <EmptyCollection text="Your conversations will appear here after your first question." />}</WorkspacePage>
}

function KnowledgePage() {
  return <WorkspacePage icon={<FiBookOpen />} eyebrow="Sources" title="Knowledge Base" description="Documents connected to your assistant."><EmptyCollection badge="Coming Soon" text="Document management will be available when the backend exposes a compatible endpoint." /></WorkspacePage>
}

function FavoritesPage() {
  return <WorkspacePage icon={<FiStar />} eyebrow="Saved" title="Favorites" description="Your most useful conversations, in one place."><EmptyCollection text="Favorite conversations will appear here." /></WorkspacePage>
}

function SavedConversationPage({ conversations, loading, language, onSend, onUpload, onInfo, onNew }: { conversations: Conversation[]; loading: boolean; language: Language; onSend: (value: string, language: Language, id?: string) => string | undefined; onUpload: () => void; onInfo: (message: Message) => void; onNew: () => void }) {
  const { id } = useParams<{ id: string }>()
  const conversation = conversations.find((item) => item.id === id)
  if (!id || !conversation) return <Navigate to="/" replace />
  return <AssistantPage conversation={conversation} loading={loading} language={language} onSend={(value, selectedLanguage) => { onSend(value, selectedLanguage, id) }} onUpload={onUpload} onInfo={onInfo} onNew={onNew} />
}

export default function App() {
  const chat = useChat()
  const navigate = useNavigate()
  const [language, setLanguage] = useState<Language>('en')
  const [menu, setMenu] = useState(false)
  const [settings, setSettings] = useState(false)
  const [upload, setUpload] = useState(false)
  const [info, setInfo] = useState<AskResponse | null>(null)
  const openConversation = (id: string) => { chat.setActiveId(id); navigate(`/conversations/${id}`) }
  const newChat = () => { chat.newChat(); navigate('/') }
  const startConversation = (value: string, selectedLanguage: Language) => {
    const id = chat.send(value, selectedLanguage)
    if (id) navigate(`/conversations/${id}`)
  }

  return <div className="app-shell"><Sidebar open={menu} onClose={() => setMenu(false)} conversations={chat.conversations} activeId={chat.activeId} onSelect={chat.setActiveId} /><main className="main"><Header language={language} onLanguage={setLanguage} onMenu={() => setMenu(true)} onNew={newChat} onSettings={() => setSettings(true)} /><Routes><Route path="/" element={<AssistantPage loading={chat.isLoading} language={language} onSend={startConversation} onUpload={() => setUpload(true)} onInfo={(message) => setInfo(message.metadata ?? null)} onNew={newChat} />} /><Route path="/conversations" element={<ConversationsPage conversations={chat.conversations} activeId={chat.activeId} onSelect={openConversation} />} /><Route path="/conversations/:id" element={<SavedConversationPage conversations={chat.conversations} loading={chat.isLoading} language={language} onSend={chat.send} onUpload={() => setUpload(true)} onInfo={(message) => setInfo(message.metadata ?? null)} onNew={newChat} />} /><Route path="/knowledge" element={<KnowledgePage />} /><Route path="/favorites" element={<FavoritesPage />} /><Route path="*" element={<Navigate to="/" replace />} /></Routes></main><SettingsDrawer open={settings} onClose={() => setSettings(false)} /><InfoDrawer data={info} onClose={() => setInfo(null)} /><UploadModal open={upload} onClose={() => setUpload(false)} /></div>
}
