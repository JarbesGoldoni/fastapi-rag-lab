/**
 * Pull complete sentences off the front of a streaming buffer so speech can start
 * before the whole answer arrives. Citation markers like [1] are removed from speech.
 */
export interface Taken {
  sentences: string[];
  rest: string;
}

const END = /([.!?…]+["')\]]*)(\s+|$)/g;

export function speakable(text: string): string {
  return text
    .replace(/\[\d{1,3}\]/g, "")
    .replace(/[*_`#>]/g, "")
    .replace(/\s+/g, " ")
    .replace(/\s+([.,!?;:…])/g, "$1")
    .trim();
}

export function takeSentences(buffer: string, { flush = false, minLength = 12 } = {}): Taken {
  const sentences: string[] = [];
  let start = 0;
  for (const match of buffer.matchAll(END)) {
    const end = (match.index ?? 0) + match[1]!.length;
    const atEnd = end === buffer.length;
    if (atEnd && !flush) break; // a "." at the very end may be a decimal point still streaming
    const sentence = speakable(buffer.slice(start, end));
    if (sentence.length >= minLength || flush) {
      if (sentence) sentences.push(sentence);
      start = end;
    }
  }
  let rest = buffer.slice(start);
  if (flush) {
    const last = speakable(rest);
    if (last) sentences.push(last);
    rest = "";
  }
  return { sentences, rest };
}
