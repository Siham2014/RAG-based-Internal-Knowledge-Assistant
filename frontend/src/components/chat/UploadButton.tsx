import { FiPaperclip } from 'react-icons/fi'
export function UploadButton({ onClick }: { onClick: () => void }) { return <button className="input-tool" onClick={onClick} type="button" aria-label="Add document"><FiPaperclip /></button> }
