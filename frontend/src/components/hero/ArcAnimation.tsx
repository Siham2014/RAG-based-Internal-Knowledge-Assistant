import { motion, useScroll, useTransform } from 'framer-motion'
export function ArcAnimation() {
  const { scrollY } = useScroll(); const opacity = useTransform(scrollY, [0, 240], [1, 0]); const filter = useTransform(scrollY, [0, 240], ['blur(0px)', 'blur(18px)'])
  return <motion.div className="arc-wrap" style={{ opacity, filter }} aria-hidden="true"><div className="arc-glow" /><div className="arc-line" /></motion.div>
}
