/**
 * Swastham voice engine — shared by the Ask tab (voice assistant) and the Voice tab (voice agent).
 *
 *   Voice.Mic       microphone + level meter (echo cancellation on, so the speaker bleeds in less)
 *   Voice.capture   records one utterance: waits for speech, stops after a pause (energy-based VAD)
 *   Voice.Speaker   reads a list of sentences aloud, fetching sentence n+1 while n plays; stop() is instant
 *   Voice.transcribe / Voice.synthesise   thin wrappers over /api/transcribe and /api/speak
 */

const Voice = (() => {
    const POLL_MS = 50;
    const SPEECH_START_MS = 200;   // above threshold this long → speech has started
    const SILENCE_END_MS = 1200;   // below threshold this long after speech → utterance ended
    const MIN_SPEECH_MS = 400;     // shorter blips (a cough, a click) are ignored
    const MAX_UTTERANCE_MS = 30000;
    const BARGE_IN_MS = 300;       // user must talk this long over the agent to interrupt it

    async function apiError(resp) {
        try {
            return (await resp.json()).error || `HTTP ${resp.status}`;
        } catch {
            return `HTTP ${resp.status}`;
        }
    }

    // --- Server calls ---

    async function transcribe(blob, language) {
        const resp = await fetch(`/api/transcribe?language=${language || "auto"}`, {
            method: "POST",
            headers: { "Content-Type": blob.type || "audio/webm" },
            body: blob,
        });
        if (!resp.ok) throw new Error(await apiError(resp));
        return resp.json();
    }

    async function synthesise(text, language) {
        const resp = await fetch("/api/speak", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text, language }),
        });
        if (!resp.ok) throw new Error(await apiError(resp));
        return URL.createObjectURL(await resp.blob());
    }

    // --- Microphone ---

    class Mic {
        async open() {
            if (this.stream) return;
            if (!navigator.mediaDevices?.getUserMedia) throw new Error("This browser cannot record audio.");
            this.stream = await navigator.mediaDevices.getUserMedia({
                audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
            });
            this.ctx = new AudioContext();
            this.analyser = this.ctx.createAnalyser();
            this.analyser.fftSize = 1024;
            this.ctx.createMediaStreamSource(this.stream).connect(this.analyser);
            this.samples = new Float32Array(this.analyser.fftSize);
            await this.calibrate();
        }

        level() {
            this.analyser.getFloatTimeDomainData(this.samples);
            let sum = 0;
            for (const s of this.samples) sum += s * s;
            return Math.sqrt(sum / this.samples.length);
        }

        async calibrate() {
            // 0.5 s of room noise sets the speech threshold, so a quiet or noisy room both work
            let total = 0, n = 0;
            const until = performance.now() + 500;
            while (performance.now() < until) {
                total += this.level();
                n += 1;
                await new Promise((r) => setTimeout(r, POLL_MS));
            }
            const noise = total / Math.max(n, 1);
            this.threshold = Math.max(noise * 3, 0.015);
            this.bargeThreshold = this.threshold * 2;
        }

        close() {
            this.stream?.getTracks().forEach((t) => t.stop());
            this.ctx?.close();
            this.stream = this.ctx = this.analyser = null;
        }
    }

    function recorderMime() {
        return ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/mp4"]
            .find((m) => MediaRecorder.isTypeSupported(m)) || "";
    }

    /**
     * Record one utterance from an open Mic.
     * Returns { done: Promise<Blob|null>, finish(), cancel() }.
     *   done resolves with the audio, or null if nobody spoke before startTimeoutMs (or on cancel).
     *   finish() ends the recording now (a second click on the mic button); cancel() discards it.
     * alreadySpeaking: the user is mid-sentence (barge-in), so don't wait for speech to start.
     */
    function capture(mic, { startTimeoutMs = 10000, alreadySpeaking = false, onSpeech } = {}) {
        const mime = recorderMime();
        let recorder, chunks, timer, settle;
        let aboveMs = 0, belowMs = 0, speechMs = 0, waitedMs = 0, totalMs = 0;
        let speaking = alreadySpeaking;

        const done = new Promise((resolve) => { settle = resolve; });

        function startRecorder() {
            chunks = [];
            recorder = new MediaRecorder(mic.stream, mime ? { mimeType: mime } : undefined);
            recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);
            recorder.start();
        }

        function end(keep) {
            clearInterval(timer);
            if (recorder.state === "inactive") return settle(null);
            recorder.onstop = () => settle(keep && chunks.length ? new Blob(chunks, { type: recorder.mimeType }) : null);
            recorder.stop();
        }

        startRecorder();
        if (speaking) onSpeech?.();
        timer = setInterval(() => {
            const loud = mic.level() > mic.threshold;
            totalMs += POLL_MS;
            if (!speaking) {
                aboveMs = loud ? aboveMs + POLL_MS : 0;
                waitedMs += POLL_MS;
                if (aboveMs >= SPEECH_START_MS) {
                    speaking = true;
                    speechMs = aboveMs;
                    onSpeech?.();
                } else if (waitedMs >= startTimeoutMs) {
                    end(false);
                }
                return;
            }
            if (loud) {
                speechMs += POLL_MS;
                belowMs = 0;
            } else {
                belowMs += POLL_MS;
            }
            if (belowMs >= SILENCE_END_MS || totalMs >= MAX_UTTERANCE_MS) {
                if (speechMs >= MIN_SPEECH_MS || alreadySpeaking) {
                    end(true);
                } else {
                    // A blip, not a question: start over and keep waiting
                    speaking = false;
                    aboveMs = belowMs = speechMs = 0;
                    recorder.onstop = null;
                    recorder.stop();
                    startRecorder();
                }
            }
        }, POLL_MS);

        return { done, finish: () => end(true), cancel: () => end(false) };
    }

    /** Calls onBargeIn once if the user talks over the agent for BARGE_IN_MS. Returns a stop function. */
    function watchForBargeIn(mic, onBargeIn) {
        let loudMs = 0;
        const timer = setInterval(() => {
            loudMs = mic.level() > mic.bargeThreshold ? loudMs + POLL_MS : 0;
            if (loudMs >= BARGE_IN_MS) {
                clearInterval(timer);
                onBargeIn();
            }
        }, POLL_MS);
        return () => clearInterval(timer);
    }

    // --- Speech output ---

    class Speaker {
        constructor() {
            this.audio = new Audio();
            this.cache = new Map(); // short fixed phrases (fillers) are synthesised once
            this.run = 0;
        }

        get speaking() {
            return this.active === this.run;
        }

        /** Speak the sentences in order. Resolves true when finished, false if stopped. */
        async play(sentences, language, { cache = false } = {}) {
            this.stop();
            const run = ++this.run;
            this.active = run;
            const fetchUrl = (text) => {
                const key = `${language}|${text}`;
                if (cache && this.cache.has(key)) return this.cache.get(key);
                const url = synthesise(text, language);
                if (cache) this.cache.set(key, url);
                return url;
            };
            const urls = sentences.map(() => null);
            const prefetch = (i) => {
                if (i < sentences.length && !urls[i]) {
                    urls[i] = fetchUrl(sentences[i]);
                    urls[i].catch(() => {}); // failures surface when the sentence is awaited
                }
            };
            try {
                for (let i = 0; i < sentences.length; i++) {
                    prefetch(i);
                    prefetch(i + 1); // next sentence synthesises while this one plays
                    const url = await urls[i];
                    if (run !== this.run) return false;
                    await this.playUrl(url, run);
                    if (!cache) URL.revokeObjectURL(url);
                    if (run !== this.run) return false;
                }
                return true;
            } finally {
                if (this.active === run) this.active = undefined;
            }
        }

        playUrl(url, run) {
            return new Promise((resolve, reject) => {
                this.audio.src = url;
                this.audio.onended = () => resolve();
                this.audio.onerror = () => reject(new Error("Audio playback failed"));
                this.stopCurrent = resolve;
                this.audio.play().catch(reject);
                if (run !== this.run) resolve();
            });
        }

        stop() {
            this.run += 1;
            this.active = undefined;
            this.audio.pause();
            this.stopCurrent?.();
        }
    }

    return { Mic, Speaker, capture, watchForBargeIn, transcribe, synthesise };
})();
