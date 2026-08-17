import { HiOutlineSparkles } from 'react-icons/hi2'
import { Link } from 'react-router-dom'

export function Logo({ compact = false, onClick }: { compact?: boolean; onClick?: () => void }) {
  return <Link className="logo" to="/" onClick={onClick} aria-label="Go to Nexus AI assistant"><span className="logo-mark"><HiOutlineSparkles /></span>{!compact && <span>Nexus<span>AI</span></span>}</Link>
}
