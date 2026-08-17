const TECHNICAL_CITATION = /\[(?=[^\[\]\r\n]*(?:\.(?:pdf|md|html?|txt)\s*:|__chunk_\d+))[^\[\]\r\n]+\]/gi

export function cleanAnswerForDisplay(answer: string): string {
  return answer
    .replace(TECHNICAL_CITATION, '')
    .replace(/[ \t]{2,}/g, ' ')
    .replace(/[ \t]+([,.;:!?])/g, '$1')
    .replace(/[ \t]+\n/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim()
}
