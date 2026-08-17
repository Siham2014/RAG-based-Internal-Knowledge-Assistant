import { motion } from 'framer-motion'
import { FiCheckCircle, FiCopy, FiInfo } from 'react-icons/fi'
import type { Message as MessageType } from '../../types/chat'
import { cleanAnswerForDisplay } from '../../utils/answerDisplay'

export function Message({ message, onInfo }: { message: MessageType; onInfo: (message: MessageType) => void }) {
  if (message.role === 'user') return <motion.div className="message-row user-row" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}><div className="user-bubble">{message.content}</div></motion.div>
  const displayAnswer = cleanAnswerForDisplay(message.content)
  const copy = () => navigator.clipboard.writeText(displayAnswer)
  return <motion.div className={`message-row assistant-row ${message.error ? 'message-error' : ''}`} initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: .35 }}><div className="assistant-badge">N</div><div className="assistant-message"><div className="message-copy">{displayAnswer}</div><div className="message-tools"><button onClick={copy} title="Copy clean answer"><FiCopy /></button>{message.metadata && <><button onClick={() => onInfo(message)} title="Response details"><FiInfo /></button><span className="verified-badge"><FiCheckCircle /> Verified</span></>}</div></div></motion.div>
}
