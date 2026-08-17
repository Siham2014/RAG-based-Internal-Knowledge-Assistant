import { useEffect, useRef } from 'react'
import type { Message as MessageType } from '../../types/chat'
import { Message } from './Message'
import { LoadingAnimation } from './LoadingAnimation'

interface Props { messages: MessageType[]; loading: boolean; onInfo: (message: MessageType) => void }
export function ChatWindow({ messages, loading, onInfo }: Props) {
  const bottom = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  return <div className="chat-window">{messages.map((message) => <Message key={message.id} message={message} onInfo={onInfo} />)}{loading && <LoadingAnimation />}<div ref={bottom} /></div>
}
