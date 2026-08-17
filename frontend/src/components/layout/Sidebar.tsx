import { AnimatePresence, motion } from 'framer-motion'
import { FiBookOpen, FiMessageSquare, FiStar, FiX } from 'react-icons/fi'
import { RiRobot2Line } from 'react-icons/ri'
import { NavLink, useNavigate } from 'react-router-dom'
import type { Conversation } from '../../types/chat'
import { Logo } from '../common/Logo'

const menus = [{ label: 'Assistant', path: '/', icon: RiRobot2Line }, { label: 'Conversations', path: '/conversations', icon: FiMessageSquare }, { label: 'Knowledge Base', path: '/knowledge', icon: FiBookOpen }, { label: 'Favorites', path: '/favorites', icon: FiStar }]

interface Props { open: boolean; onClose: () => void; conversations: Conversation[]; activeId: string | null; onSelect: (id: string) => void }
export function Sidebar({ open, onClose, conversations, activeId, onSelect }: Props) {
  const navigate = useNavigate()

  const selectConversation = (id: string) => {
    onSelect(id)
    navigate(`/conversations/${id}`)
    onClose()
  }

  return <>
    <AnimatePresence>{open && <motion.button aria-label="Close menu" className="sidebar-backdrop" onClick={onClose} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} />}</AnimatePresence>
    <aside className={`sidebar ${open ? 'is-open' : ''}`}>
      <div className="sidebar-head"><Logo onClick={onClose} /><button className="icon-button mobile-only" onClick={onClose} aria-label="Close menu"><FiX /></button></div>
      <nav className="main-nav">{menus.map(({ label, path, icon: Icon }) => <NavLink end={path === '/'} to={path} onClick={onClose} className={({ isActive }) => isActive ? 'active' : ''} key={path}><Icon /><span>{label}</span></NavLink>)}</nav>
      {conversations.length > 0 && <div className="recent"><p>Recent</p>{conversations.slice(0, 5).map((chat) => <button key={chat.id} className={chat.id === activeId ? 'selected' : ''} onClick={() => selectConversation(chat.id)}><span>{chat.title}</span></button>)}</div>}
      <div className="sidebar-profile"><div className="avatar">NA</div><div><strong>Nexus workspace</strong><span>Private assistant</span></div><i /></div>
    </aside>
  </>
}
