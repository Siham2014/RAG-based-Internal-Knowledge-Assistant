import { FiMenu, FiPlus, FiSettings } from 'react-icons/fi'
import { Logo } from '../common/Logo'
import { LanguageSelector } from '../chat/LanguageSelector'
import type { Language } from '../../types/chat'

interface Props { language: Language; onLanguage: (value: Language) => void; onMenu: () => void; onNew: () => void; onSettings: () => void }
export function Header({ language, onLanguage, onMenu, onNew, onSettings }: Props) {
  return <header className="header"><div className="header-left"><button className="icon-button menu-button" onClick={onMenu} aria-label="Open menu"><FiMenu /></button><div className="header-logo"><Logo /></div></div><div className="header-actions"><LanguageSelector value={language} onChange={onLanguage} /><button className="action-button desktop-action" onClick={onNew}><FiPlus /> New chat</button><button className="icon-button" onClick={onSettings} aria-label="Settings"><FiSettings /></button></div></header>
}
