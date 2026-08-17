import { motion } from 'framer-motion'
export function RobotHero() {
  return <motion.div className="robot-wrap" animate={{ y: [0, -9, 0], rotate: [-1.2, 1.2, -1.2] }} transition={{ duration: 5.5, ease: 'easeInOut', repeat: Infinity }}><div className="robot-aura" /><img src="/assets/nexus-robot.png" alt="Nexus AI assistant" /></motion.div>
}
