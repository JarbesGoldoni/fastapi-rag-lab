/** Browser audio helpers: recording to 16 kHz mono PCM for Whisper, and gapless playback. */

const WHISPER_RATE = 16_000;

export async function decodeToMono16k(blob: Blob): Promise<Float32Array> {
  const context = new AudioContext({ sampleRate: WHISPER_RATE });
  try {
    const buffer = await context.decodeAudioData(await blob.arrayBuffer());
    if (buffer.numberOfChannels === 1) return buffer.getChannelData(0).slice();
    const left = buffer.getChannelData(0);
    const right = buffer.getChannelData(1);
    return left.map((value, i) => (value + (right[i] ?? 0)) / 2);
  } finally {
    await context.close();
  }
}

export interface Recorder {
  stop: () => Promise<Blob>;
  cancel: () => void;
}

export async function startRecording(): Promise<Recorder> {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
  });
  const recorder = new MediaRecorder(stream);
  const chunks: Blob[] = [];
  recorder.ondataavailable = (event) => chunks.push(event.data);
  recorder.start();
  const release = () => stream.getTracks().forEach((track) => track.stop());
  return {
    stop: () =>
      new Promise((resolve) => {
        recorder.onstop = () => {
          release();
          resolve(new Blob(chunks, { type: recorder.mimeType }));
        };
        recorder.stop();
      }),
    cancel: () => {
      recorder.onstop = release;
      recorder.stop();
    },
  };
}

/** Plays PCM clips back to back without gaps; stop() silences everything queued. */
export function createPlayer() {
  let context: AudioContext | null = null;
  let endsAt = 0;
  let sources: AudioBufferSourceNode[] = [];

  return {
    play(samples: Float32Array, sampleRate: number): Promise<void> {
      context ??= new AudioContext();
      const buffer = context.createBuffer(1, samples.length, sampleRate);
      buffer.copyToChannel(samples as Float32Array<ArrayBuffer>, 0);
      const source = context.createBufferSource();
      source.buffer = buffer;
      source.connect(context.destination);
      const startAt = Math.max(context.currentTime + 0.05, endsAt);
      source.start(startAt);
      endsAt = startAt + buffer.duration;
      sources.push(source);
      return new Promise((resolve) => {
        source.onended = () => {
          sources = sources.filter((s) => s !== source);
          resolve();
        };
      });
    },
    stop() {
      sources.forEach((s) => s.stop());
      sources = [];
      endsAt = 0;
    },
    get playing() {
      return sources.length > 0;
    },
  };
}
