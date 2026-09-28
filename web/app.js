/**
 * Swastham Health RAG — frontend logic
 */

const $ = (id) => document.getElementById(id);

const tabs = document.querySelectorAll(".nav-tab");
const views = { ask: $("view-ask"), compare: $("view-compare"), voice: $("view-voice") };
const langPills = document.querySelectorAll("#view-ask .lang-pill");
const errorBanner = $("error-banner");

let selectedLang = "auto"; // "auto", "en", "hi", "hinglish"

const ANSWER_BADGES = { en: "English Answer", hi: "हिन्दी उत्तर (Hindi Answer)", hinglish: "Hinglish Answer" };

const CHECK_LABELS = {
    retrieval_overlap: "Same evidence retrieved",
    numbers_match: "Numbers identical",
    cautions_preserved: "Safety cautions preserved",
    recommendation_match: "Recommendations preserved",
    content_coverage: "Same content (back-translation)",
};

// --- Tabs ---
tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
        const target = tab.dataset.tab;
        tabs.forEach((t) => t.classList.toggle("active", t.dataset.tab === target));
        Object.entries(views).forEach(([name, el]) => {
            el.classList.toggle("active", name === target);
            el.classList.toggle("hidden", name !== target);
        });
        speaker.stop();
        document.dispatchEvent(new CustomEvent("tabchange", { detail: target }));
    });
});

// --- Language pills ---
function setLang(lang) {
    selectedLang = lang;
    langPills.forEach((p) => p.classList.toggle("active", p.dataset.lang === lang));
}
langPills.forEach((pill) => pill.addEventListener("click", () => setLang(pill.dataset.lang)));

// --- Sample chips ---
document.querySelectorAll(".chip[data-query]").forEach((chip) => {
    chip.addEventListener("click", () => {
        $("query-input").value = chip.dataset.query;
        setLang("auto");
        $("query-input").focus();
    });
});
document.querySelectorAll(".compare-pair-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
        $("compare-en").value = chip.dataset.en;
        $("compare-hi").value = chip.dataset.hi;
        $("compare-hg").value = chip.dataset.hg || "";
    });
});

// --- Status & errors ---
async function checkStatus() {
    try {
        const data = await (await fetch("/api/status")).json();
        $("key-status-dot").classList.toggle("live", data.has_key);
        $("llm-status-text").textContent = data.has_key ? data.model : "No LLM key";
        if (!data.has_key) showError("GROQ_API_KEY is not set in .env. Add it and restart the server.");
        voiceAvailable = !!data.voice;
        if (!voiceAvailable) {
            $("btn-mic").disabled = true;
            $("btn-mic").title = "Voice needs GROQ_API_KEY and edge-tts (pip install -r requirements.txt)";
        }
    } catch {
        showError("Cannot reach the server.");
    }
}

function showError(message) {
    errorBanner.textContent = message;
    errorBanner.classList.remove("hidden");
}

function clearError() {
    errorBanner.classList.add("hidden");
}

async function postJson(url, payload) {
    const resp = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    });
    const data = await resp.json();
    if (data.error) throw new Error(data.error);
    return data;
}

async function withSpinner(button, spinner, busyText, fn) {
    const label = button.querySelector(".btn-text");
    const idleText = label.textContent;
    button.disabled = true;
    spinner.classList.remove("hidden");
    label.textContent = busyText;
    clearError();
    try {
        await fn();
    } catch (err) {
        showError(`Error: ${err.message}`);
    } finally {
        button.disabled = false;
        spinner.classList.add("hidden");
        label.textContent = idleText;
    }
}

// --- Ask ---
$("btn-ask").addEventListener("click", () => handleAsk());
$("query-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleAsk();
    }
});

/**
 * spoken: the question came from the mic. It is a transcript (Hinglish speech is written in
 * Devanagari), so the pipeline detects the input language from the text and the language pill
 * only chooses the reply language.
 */
function handleAsk({ spoken = false } = {}) {
    const q = $("query-input").value.trim();
    if (!q) return;
    speaker.stop();
    $("ask-results").classList.add("hidden");
    withSpinner($("btn-ask"), $("ask-spinner"), "Processing…", async () => {
        const payload = { query: q };
        if (selectedLang !== "auto") payload[spoken ? "reply_language" : "language"] = selectedLang;
        renderAskResult(await postJson("/api/query", payload));
    });
}

// --- Voice assistant: speak a question, hear the answer ---
const speaker = new Voice.Speaker();
const askMic = new Voice.Mic();
let voiceAvailable = false;
let micCapture = null;
let lastSpeech = null; // { sentences, language } of the answer on screen

function micErrorMessage(err) {
    return err.name === "NotAllowedError"
        ? "Microphone access was blocked. Allow the microphone for this site (lock icon in the address bar) and try again."
        : `Microphone error: ${err.message}`;
}

function setMicState(state, text = "") {
    $("btn-mic").dataset.state = state;
    $("mic-status").textContent = text;
    $("mic-status").classList.toggle("hidden", !text);
}

$("btn-mic").addEventListener("click", async () => {
    if (micCapture) {
        micCapture.finish(); // second click: stop recording now
        return;
    }
    speaker.stop();
    clearError();
    try {
        setMicState("busy", "Getting the microphone ready…");
        await askMic.open();
    } catch (err) {
        setMicState("idle");
        showError(micErrorMessage(err));
        return;
    }
    setMicState("recording", "Listening… stops when you pause, or click the mic");
    micCapture = Voice.capture(askMic, { startTimeoutMs: 8000 });
    const blob = await micCapture.done;
    micCapture = null;
    askMic.close();
    if (!blob) {
        setMicState("idle");
        showError("No speech heard. Click the mic and ask your question.");
        return;
    }
    setMicState("busy", "Transcribing…");
    try {
        const t = await Voice.transcribe(blob, selectedLang === "auto" ? null : selectedLang);
        $("query-input").value = t.text;
        $("auto-read").checked = true;
        setMicState("idle");
        handleAsk({ spoken: true });
    } catch (err) {
        setMicState("idle");
        showError(`Could not transcribe: ${err.message}`);
    }
});

async function readAnswerAloud() {
    if (!lastSpeech) return;
    $("btn-listen").textContent = "⏹ Stop";
    try {
        await speaker.play(lastSpeech.sentences, lastSpeech.language);
    } catch (err) {
        showError(`Could not read the answer aloud: ${err.message}`);
    } finally {
        $("btn-listen").textContent = "🔊 Listen";
    }
}

$("btn-listen").addEventListener("click", () => {
    if (speaker.speaking) speaker.stop();
    else readAnswerAloud();
});

function renderAskResult(data) {
    const isHindi = data.output_language === "hi";
    const isTranslated = isHindi || data.output_language === "hinglish";
    const parity = data.hindi_parity || data.hinglish_parity;
    const back = data.hindi_back_translation || data.hinglish_back_translation;
    const langName = isHindi ? "Hindi" : "Hinglish";
    $("answer-lang-badge").textContent = ANSWER_BADGES[data.output_language] || "Answer";
    $("answer-body").className = isHindi ? "answer-content hindi-text" : "answer-content";
    $("answer-body").innerHTML = formatMarkdown(data.final_answer);

    const v = data.validation;
    const chunkIds = (data.retrieved_chunks || []).map((c) => `#${c.id}`).join(", ");
    $("answer-meta").textContent =
        `Chunks ${chunkIds} · validation ${v.passed ? "passed" : "FAILED"}${data.answer_reused ? " · reused stored answer" : ""}`;

    const plan = data.answer_plan || {};
    $("trace-body").innerHTML = `
        ${data.standard_hindi_query ? `<div class="evidence-item"><div class="evidence-title">Hinglish normalisation</div>
            <div class="evidence-text">Normalised: ${escapeHtml(data.normalised_hinglish)}
                <br>Standard Hindi: <span class="hindi-text">${escapeHtml(data.standard_hindi_query)}</span>
                ${(data.protected_terms || []).length ? `<br><small>Medical terms kept verbatim: ${escapeHtml(data.protected_terms.join(", "))}</small>` : ""}
                ${data.hinglish_conversion_reused ? "<br><small>♻ This wording was converted before; the stored conversion was reused.</small>" : ""}</div></div>` : ""}
        <div class="evidence-item"><div class="evidence-title">Canonical question used for retrieval</div>
            <div class="evidence-text">${escapeHtml(data.english_query)}
                ${data.translated_query !== data.user_query ? `<br><small>Translated from your question as: ${escapeHtml(data.translated_query)}</small>` : ""}
                <br><small>${data.answer_reused
                    ? "♻ Same canonical question was answered before: the stored answer and chunks were reused."
                    : "New canonical question: answer generated and stored."}</small></div></div>
        <div class="evidence-item"><div class="evidence-title">Answer Plan</div>
            ${planList("Required points", plan.required_points)}
            ${planList("Important conditions", plan.important_conditions)}
            ${planList("Restrictions", plan.restrictions)}</div>
        <div class="evidence-item"><div class="evidence-title">Validation</div>
            ${checkList(v.checks)}</div>
        ${isTranslated ? `<div class="evidence-item"><div class="evidence-title">Canonical English answer (source of the ${langName} text)</div>
            <div class="evidence-text">${formatMarkdown(data.canonical_answer)}</div></div>` : ""}
        ${parity ? `<div class="evidence-item"><div class="evidence-title">${langName} parity check
            (${parity.parity_passed ? "passed" : "FAILED"}${data.translation_reused ? ", stored translation reused" : ""})</div>
            ${checkList(parity.checks)}
            <div class="evidence-text"><small>Back-translation:</small><br>${formatMarkdown(back)}</div></div>` : ""}
    `;

    const chunks = data.retrieved_chunks || [];
    $("chunks-summary").textContent = `Retrieved evidence (${chunks.length} chunks)`;
    $("chunks-body").innerHTML = chunks.map((c) => `
        <div class="evidence-item">
            <div class="evidence-title">Chunk #${c.id} · score ${c.score.toFixed(2)}${c.safety_critical ? " · ⚠ safety-critical" : ""}
                — ${escapeHtml(c.heading)}</div>
            <div class="evidence-text">${escapeHtml(c.text.slice(0, 320))}…</div>
        </div>`).join("");

    $("ask-results").classList.remove("hidden");
    $("ask-results").scrollIntoView({ behavior: "smooth", block: "nearest" });

    lastSpeech = { sentences: data.speech_sentences || [], language: data.output_language };
    $("btn-listen").classList.toggle("hidden", !voiceAvailable || !lastSpeech.sentences.length);
    if (voiceAvailable && $("auto-read").checked) readAnswerAloud();
}

function planList(title, items) {
    if (!items || !items.length) return "";
    return `<div class="plan-group"><strong>${title}</strong><ul>${
        items.map((i) => `<li>${escapeHtml(typeof i === "string" ? i : JSON.stringify(i))}</li>`).join("")
    }</ul></div>`;
}

function checkList(checks) {
    return `<ul class="check-list">${checks.map((c) =>
        `<li class="${c.passed ? "ok" : "bad"}">${c.passed ? "✓" : "✗"} ${escapeHtml(c.name)}: ${escapeHtml(c.details)}</li>`
    ).join("")}</ul>`;
}

// --- Compare ---
$("btn-compare").addEventListener("click", handleCompare);

function handleCompare() {
    const enQ = $("compare-en").value.trim();
    const hiQ = $("compare-hi").value.trim();
    const hgQ = $("compare-hg").value.trim();
    if (!enQ || !hiQ) {
        showError("Please provide both an English and a Hindi question (Hinglish is optional).");
        return;
    }
    const baseline = $("baseline-toggle").checked;
    const payload = { english_query: enQ, hindi_query: hiQ, baseline };
    if (hgQ) payload.hinglish_query = hgQ;
    $("compare-results").classList.add("hidden");
    withSpinner($("btn-compare"), $("compare-spinner"), baseline ? "Running both pipelines…" : "Analyzing…", async () => {
        renderCompareResult(await postJson("/api/compare", payload));
    });
}

function renderCompareResult(data) {
    const dot = $("overlap-dot");
    if (data.status === "Question Intent Mismatch") {
        dot.className = "overlap-dot bad";
        const which = data.hinglish_translated ? "Hinglish" : "Hindi";
        $("overlap-text").textContent =
            `Question Intent Mismatch (similarity ${data.query_similarity}). The ${which} question translates to: "${data.hinglish_translated || data.hindi_translated}"`;
        $("scorecards").innerHTML = "";
        $("dual-answers").classList.add("hidden");
        $("compare-results").classList.remove("hidden");
        return;
    }

    $("dual-answers").classList.remove("hidden");
    const hasHg = !!data.hinglish;
    const passed = data.parity.parity_passed && (!hasHg || data.hinglish_parity.parity_passed);
    dot.className = `overlap-dot ${passed ? "" : "bad"}`;
    const same = data.same_canonical_question && (!hasHg || data.same_canonical_question_hinglish);
    const languages = hasHg ? "English, Hindi and Hinglish" : "English and Hindi";
    $("overlap-text").textContent = (passed
        ? `Parity passed: the ${languages} answers carry the same information. `
        : "Parity check flagged differences. See the scorecards. ")
        + (same
            ? `All questions normalised to "${data.english.english_query}", so they used the same chunks and the same canonical answer.`
            : `The questions normalised differently (EN → "${data.english.english_query}" | HI → "${data.hindi.english_query}"`
              + (hasHg ? ` | HG → "${data.hinglish.english_query}"` : "") + "). "
              + "The translated answers still come from the English canonical answer; on the Ask tab a question that normalised differently would get its own answer.");

    let cards = scorecard("Hindi: canonical pipeline (new)", data.parity);
    if (hasHg) cards += scorecard("Hinglish: canonical pipeline (new)", data.hinglish_parity);
    if (data.baseline) cards += scorecard("Hindi baseline (before)", data.baseline.parity);
    if (data.baseline && data.baseline.hinglish_parity) cards += scorecard("Hinglish baseline (before)", data.baseline.hinglish_parity);
    $("scorecards").innerHTML = cards;

    $("compare-en-answer").innerHTML = formatMarkdown(data.english.final_answer);
    $("compare-hi-answer").innerHTML = formatMarkdown(data.hindi.final_answer);
    $("compare-hi-back").innerHTML = formatMarkdown(data.hindi_back_translation);
    $("hinglish-column").classList.toggle("hidden", !hasHg);
    if (hasHg) {
        $("compare-hg-answer").innerHTML = formatMarkdown(data.hinglish.final_answer);
        $("compare-hg-back").innerHTML = formatMarkdown(data.hinglish_back_translation);
    }

    const baseCol = $("baseline-column");
    baseCol.classList.toggle("hidden", !data.baseline);
    const columns = 2 + (hasHg ? 1 : 0) + (data.baseline ? 1 : 0);
    $("dual-answers").classList.toggle("three", columns === 3);
    $("dual-answers").classList.toggle("four", columns === 4);
    if (data.baseline) $("compare-base-answer").innerHTML = formatMarkdown(data.baseline.hindi.final_answer);

    $("compare-results").classList.remove("hidden");
    $("compare-results").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function scorecard(title, parity) {
    const rows = parity.checks.map((c) => `
        <tr class="${c.passed ? "ok" : "bad"}">
            <td>${c.passed ? "✓" : "✗"}</td>
            <td>${CHECK_LABELS[c.name] || c.name}</td>
            <td>${escapeHtml(c.details)}</td>
        </tr>`).join("");
    return `<div class="scorecard">
        <div class="scorecard-title">${title}: <span class="${parity.parity_passed ? "ok" : "bad"}">${parity.parity_passed ? "PASS" : "FAIL"}</span></div>
        <div class="table-wrap"><table>${rows}</table></div>
    </div>`;
}

// --- Text utilities ---
function formatMarkdown(text) {
    if (!text) return "";
    let html = escapeHtml(text);
    html = html.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/^\s*(?:[•\-\*]|\d+\.)\s+(.+)$/gm, "<li>$1</li>");
    html = html.replace(/((?:<li>.*<\/li>\s*)+)/g, "<ul>$1</ul>");
    return html.split(/\n\n+/).map((p) => {
        p = p.trim();
        return p.startsWith("<ul>") ? p : `<p>${p.replace(/\n/g, "<br>")}</p>`;
    }).join("");
}

function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str ?? "";
    return div.innerHTML;
}

checkStatus();
