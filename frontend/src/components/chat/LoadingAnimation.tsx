import { motion } from 'framer-motion'
export function LoadingAnimation() { return <motion.div className="message-row assistant-row" initial={{ opacity: 0 }} animate={{ opacity: 1 }}><div className="assistant-badge">N</div><div className="loading-card"><div className="thinking"><i /><i /><i /></div><div className="shimmer-line" /><div className="shimmer-line short" /></div></motion.div> }
