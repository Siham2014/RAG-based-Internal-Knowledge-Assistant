import { useState } from 'react'
import { FiBookOpen, FiCpu, FiDatabase, FiMonitor, FiSettings, FiSliders, FiTool } from 'react-icons/fi'
import { DrawerShell } from './DrawerShell'

const tabs = [{ id: 'general', label: 'General', icon: FiSettings }, { id: 'provider', label: 'AI Provider', icon: FiCpu }, { id: 'generation', label: 'Generation', icon: FiSliders }, { id: 'knowledge', label: 'Knowledge Base', icon: FiBookOpen }, { id: 'retrieval', label: 'Retrieval', icon: FiDatabase }, { id: 'appearance', label: 'Appearance', icon: FiMonitor }, { id: 'system', label: 'System', icon: FiTool }]
export function SettingsDrawer({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [active, setActive] = useState('general'); const selected = tabs.find((tab) => tab.id === active)!
  return <DrawerShell open={open} onClose={onClose} title="Settings" subtitle="Manage your assistant preferences"><div className="settings-tabs">{tabs.map(({ id, label, icon: Icon }) => <button key={id} onClick={() => setActive(id)} className={active === id ? 'active' : ''}><Icon />{label}</button>)}</div><div className="settings-panel"><span className="coming-pill">Coming Soon</span><h3>{selected.label}</h3><p>These preferences will be available when the backend exposes a compatible configuration endpoint.</p><div className="fake-setting"><div><strong>Default preferences</strong><small>Managed by your workspace administrator</small></div><span className="fake-toggle" /></div><div className="fake-setting"><div><strong>Workspace configuration</strong><small>Read-only in this version</small></div><span className="locked">Locked</span></div></div></DrawerShell>
}
