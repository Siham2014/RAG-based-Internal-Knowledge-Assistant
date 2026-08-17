import { FiEdit3 } from 'react-icons/fi'
export function NewChatButton({ onClick }: { onClick: () => void }) { return <button className="new-chat-fab" onClick={onClick} aria-label="New chat"><FiEdit3 /></button> }
