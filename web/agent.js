/**
 * Swastham voice agent (Voice tab): a hands-free conversation loop.
 *
 *   listening → (utterance) → thinking → speaking → listening …
 *
 * Every answer comes from /api/query, the same validated, parity-checked pipeline as the
 * Ask tab; the agent only adds speech in/out and a short memory of the conversation so a
 * follow-up ("and if I'm pregnant?") is rewritten into a standalone question on the server.
 * Talking over the agent (barge-in) stops it and starts listening.
 */

(() => {
    const HISTORY_TURNS = 3;
    const IDLE_TIMEOUT_MS = 30000;

    const FILLER = {
        en: "One moment, let me check that.",
        hi: "एक क्षण, मैं जाँच कर रही हूँ।",
        hinglish: "Ek second, main check kar rahi hoon.",
    };
    // Spoken when the pipeline cannot give a verified answer (fail closed: nothing unverified is read out)
    const NO_VERIFIED_ANSWER = {
        en: "I couldn't give a verified answer to that. Please try asking again, or check with your doctor.",
        hi: "मैं इसका सत्यापित उत्तर नहीं दे पाई। कृपया दोबारा पूछें, या अपने डॉक्टर से सलाह लें।",
        hinglish: "Main iska verified jawab nahi de payi. Kripya dobara poochhiye, ya apne doctor se salah lijiye.",
    };
    const STATE_TEXT = {
        idle: "Press to start talking",
        starting: "Getting the microphone ready…",
        listening: "Listening…",
        hearing: "Listening… (pause when you're done)",
        thinking: "Checking the knowledge base…",
        speaking: "Speaking… talk to interrupt",
    };

    const mic = new Voice.Mic();
    let replyLang = "auto";
    let active = false;
    let capture = null;
    let history = []; // [{ question: canonical English question, answer: canonical English answer }]

    const orb = $("voice-orb");
    const startBtn = $("btn-voice-start");
    const chat = $("voice-chat");

    // --- UI ---

    function setState(state) {
        orb.dataset.state = state;
        $("voice-state").textContent = STATE_TEXT[state];
        orb.setAttribute("aria-label", active ? "End conversation" : "Start conversation");
        startBtn.textContent = active ? "End conversation" : "Start conversation";
        startBtn.classList.toggle("ending", active);
    }

    function addBubble(role, html, extraClass = "") {
        $("voice-empty")?.remove();
        const el = document.createElement("div");
        el.className = `voice-bubble ${role} ${extraClass}`.trim();
        el.innerHTML = html;
        chat.appendChild(el);
        el.scrollIntoView({ behavior: "smooth", block: "nearest" });
        return el;
    }

    function addNote(text) {
        addBubble("note", escapeHtml(text));
    }

    function verified(trace) {
        const parity = trace.hindi_parity || trace.hinglish_parity;
        return trace.validation?.passed && (!parity || parity.parity_passed);
    }

    function agentBubble(trace) {
        const parity = trace.hindi_parity || trace.hinglish_parity;
        const ok = verified(trace);
        const lang = trace.output_language;
        return addBubble("agent", `
            <div class="bubble-meta">
                <span class="verify-badge ${ok ? "ok" : "warn"}">${ok ? "✓ verified" : "⚠ check the trace"}</span>
                ${trace.answer_reused ? '<span class="source-tag">stored answer</span>' : ""}
            </div>
            ${trace.standalone_query ? `<div class="followup-note">Follow-up understood as: “${escapeHtml(trace.standalone_query)}”</div>` : ""}
            <div class="answer-content ${lang === "hi" ? "hindi-text" : ""}">${formatMarkdown(trace.final_answer)}</div>
            <details class="evidence-accordion">
                <summary>Trace: question, validation${parity ? ", parity" : ""}</summary>
                <div class="evidence-list">
                    <div class="evidence-item"><div class="evidence-title">Canonical question</div>
                        <div class="evidence-text">${escapeHtml(trace.english_query)}</div></div>
                    <div class="evidence-item"><div class="evidence-title">Validation</div>${checkList(trace.validation.checks)}</div>
                    ${parity ? `<div class="evidence-item"><div class="evidence-title">Parity</div>${checkList(parity.checks)}</div>` : ""}
                </div>
            </details>`);
    }

    // --- Speech helpers ---

    function spokenLang(transcriptLang) {
        return replyLang === "auto" ? transcriptLang : replyLang;
    }

    /** Speak while listening for barge-in. Resolves "done", "interrupted" or "stopped". */
    async function speakInterruptible(sentences, language) {
        let interrupted = false;
        const stopWatching = Voice.watchForBargeIn(mic, () => {
            interrupted = true;
            speaker.stop();
        });
        try {
            const finished = await speaker.play(sentences, language);
            return finished ? "done" : interrupted ? "interrupted" : "stopped";
        } finally {
            stopWatching();
        }
    }

    // --- Conversation loop ---

    async function start() {
        clearError();
        if (!voiceAvailable) {
            showError("Voice needs GROQ_API_KEY in .env and edge-tts installed (pip install -r requirements.txt).");
            return;
        }
        active = true;
        setState("starting");
        try {
            await mic.open();
        } catch (err) {
            active = false;
            setState("idle");
            showError(micErrorMessage(err));
            return;
        }
        loop();
    }

    function stop(note) {
        if (!active) return;
        active = false;
        capture?.cancel();
        speaker.stop();
        mic.close();
        setState("idle");
        if (note) addNote(note);
    }

    async function loop() {
        let bargedIn = false;
        while (active) {
            setState(bargedIn ? "hearing" : "listening");
            capture = Voice.capture(mic, {
                startTimeoutMs: IDLE_TIMEOUT_MS,
                alreadySpeaking: bargedIn,
                onSpeech: () => active && setState("hearing"),
            });
            const blob = await capture.done;
            capture = null;
            bargedIn = false;
            if (!active) return;
            if (!blob) {
                stop("No one spoke for 30 seconds, so the conversation ended.");
                return;
            }

            setState("thinking");
            let transcript;
            try {
                transcript = await Voice.transcribe(blob, null);
            } catch (err) {
                if (!/no speech/i.test(err.message)) addNote(`Could not transcribe: ${err.message}`);
                continue;
            }
            if (!active) return;
            addBubble("user", escapeHtml(transcript.text), transcript.language === "hi" ? "hindi-text" : "");

            const lang = spokenLang(transcript.language);
            const filler = speaker.play([FILLER[lang]], lang, { cache: true }).catch(() => {});
            const payload = { query: transcript.text, history };
            if (replyLang !== "auto") payload.reply_language = replyLang;

            let trace;
            try {
                trace = await postJson("/api/query", payload);
            } catch (err) {
                if (!active) return;
                addBubble("agent", `<div class="bubble-meta"><span class="verify-badge warn">⚠ no verified answer</span></div>
                    <div class="evidence-text">${escapeHtml(err.message)}</div>`, "failed");
                await filler;
                setState("speaking");
                bargedIn = (await speakInterruptible([NO_VERIFIED_ANSWER[lang]], lang)) === "interrupted";
                continue;
            }
            if (!active) return;

            agentBubble(trace);
            history = [...history, { question: trace.english_query, answer: trace.canonical_answer }].slice(-HISTORY_TURNS);

            await filler;
            if (!active) return;
            setState("speaking");
            try {
                bargedIn = (await speakInterruptible(trace.speech_sentences, trace.output_language)) === "interrupted";
            } catch (err) {
                addNote(`Could not read the answer aloud: ${err.message}`);
            }
        }
    }

    // --- Controls ---

    function toggle() {
        if (active) stop();
        else start();
    }

    orb.addEventListener("click", toggle);
    startBtn.addEventListener("click", toggle);

    $("btn-voice-clear").addEventListener("click", () => {
        history = [];
        addNote("Memory cleared: the next question starts a new conversation.");
    });

    document.querySelectorAll("#voice-lang .lang-pill").forEach((pill) => {
        pill.addEventListener("click", () => {
            replyLang = pill.dataset.lang;
            document.querySelectorAll("#voice-lang .lang-pill")
                .forEach((p) => p.classList.toggle("active", p === pill));
        });
    });

    // Leaving the Voice tab ends the conversation (and releases the microphone)
    document.addEventListener("tabchange", (e) => {
        if (e.detail !== "voice") stop();
    });

    setState("idle");
})();
