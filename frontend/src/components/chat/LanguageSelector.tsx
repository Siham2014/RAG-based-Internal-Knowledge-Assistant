import { useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { FiCheck, FiChevronDown, FiGlobe } from 'react-icons/fi'
import type { Language } from '../../types/chat'

const options: { code: Language; label: string; native: string; disabled?: boolean }[] = [
  { code: 'en', label: 'English', native: 'EN' }, { code: 'fr', label: 'Français', native: 'FR' }, { code: 'ar', label: 'العربية', native: 'AR', disabled: true },
]
export function LanguageSelector({ value, onChange }: { value: Language; onChange: (value: Language) => void }) {
  const [open, setOpen] = useState(false); const selected = options.find((item) => item.code === value)!
  return <div className="language"><button className="language-trigger" onClick={() => setOpen(!open)}><FiGlobe /><span>{selected.native}</span><FiChevronDown /></button><AnimatePresence>{open && <motion.div className="language-menu" initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }}>{options.map((item) => <button key={item.code} disabled={item.disabled} onClick={() => { onChange(item.code); setOpen(false) }}><span>{item.label}{item.disabled && <small>Coming Soon</small>}</span>{value === item.code && <FiCheck />}</button>)}</motion.div>}</AnimatePresence></div>
}
