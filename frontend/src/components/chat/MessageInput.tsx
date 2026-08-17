import { FormEvent, KeyboardEvent, useRef, useState } from 'react'
import { FiArrowUp } from 'react-icons/fi'
import { UploadButton } from './UploadButton'

interface Props { onSend: (value: string) => void; onUpload: () => void; disabled: boolean }
export function MessageInput({ onSend, onUpload, disabled }: Props) {
  const [value, setValue] = useState(''); const area = useRef<HTMLTextAreaElement>(null)
  const submit = (event?: FormEvent) => { event?.preventDefault(); if (!value.trim() || disabled) return; onSend(value); setValue(''); if (area.current) area.current.style.height = 'auto' }
  const key = (event: KeyboardEvent<HTMLTextAreaElement>) => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); submit() } }
  return <form className="composer" onSubmit={submit}><textarea ref={area} value={value} rows={1} placeholder="Ask anything about your knowledge base..." aria-label="Message" onChange={(e) => { setValue(e.target.value); e.target.style.height = 'auto'; e.target.style.height = `${Math.min(e.target.scrollHeight, 160)}px` }} onKeyDown={key} /><div className="composer-bottom"><div><UploadButton onClick={onUpload} /><span className="tool-label">Add document</span></div><button className="send-button" disabled={!value.trim() || disabled} type="submit" aria-label="Send message"><FiArrowUp /></button></div></form>
}
