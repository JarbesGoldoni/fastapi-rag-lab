import { useEffect, useRef, useState } from "react";
import { MicIcon, SendIcon, StopIcon } from "./Icons";

interface Props {
  busy: boolean;
  disabled: boolean;
  placeholder: string;
  listening: "idle" | "recording" | "transcribing";
  micProgress: number | null;
  onSend: (text: string) => void;
  onStop: () => void;
  onMic: () => void;
  transcript: string | null;
}

export function Composer(props: Props) {
  const [text, setText] = useState("");
  const area = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (props.transcript) setText((t) => (t ? `${t} ${props.transcript}` : props.transcript!));
  }, [props.transcript]);

  useEffect(() => {
    const el = area.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 180)}px`;
  }, [text]);

  const submit = () => {
    const question = text.trim();
    if (!question || props.busy || props.disabled) return;
    props.onSend(question);
    setText("");
  };

  const micLabel =
    props.listening === "recording"
      ? "Stop recording"
      : props.listening === "transcribing"
        ? "Transcribing…"
        : props.micProgress !== null
          ? `Loading speech model ${props.micProgress}%`
          : "Speak your question";

  return (
    <div className="composer">
      <button
        className={`icon-button mic ${props.listening}`}
        onClick={props.onMic}
        disabled={props.listening === "transcribing"}
        title={micLabel}
        aria-label={micLabel}
      >
        <MicIcon />
        {props.micProgress !== null && props.listening === "idle" && (
          <span className="ring" style={{ ["--p" as string]: `${props.micProgress}%` }} />
        )}
      </button>
      <textarea
        ref={area}
        rows={1}
        value={text}
        placeholder={props.listening === "recording" ? "Listening…" : props.placeholder}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            submit();
          }
        }}
        disabled={props.disabled}
      />
      {props.busy ? (
        <button className="icon-button send" onClick={props.onStop} title="Stop" aria-label="Stop">
          <StopIcon />
        </button>
      ) : (
        <button
          className="icon-button send"
          onClick={submit}
          disabled={!text.trim() || props.disabled}
          title="Send"
          aria-label="Send"
        >
          <SendIcon />
        </button>
      )}
    </div>
  );
}
