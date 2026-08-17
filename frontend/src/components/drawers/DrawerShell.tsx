import type { ReactNode } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { FiX } from 'react-icons/fi'

export function DrawerShell({ open, onClose, title, subtitle, children }: { open: boolean; onClose: () => void; title: string; subtitle?: string; children: ReactNode }) {
  return <AnimatePresence>{open && <><motion.button className="overlay" aria-label="Close panel" onClick={onClose} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} /><motion.aside className="drawer" initial={{ x: '100%' }} animate={{ x: 0 }} exit={{ x: '100%' }} transition={{ type: 'spring', damping: 30, stiffness: 280 }}><div className="drawer-head"><div><h2>{title}</h2>{subtitle && <p>{subtitle}</p>}</div><button className="icon-button" onClick={onClose} aria-label="Close"><FiX /></button></div><div className="drawer-body">{children}</div></motion.aside></>}</AnimatePresence>
}
