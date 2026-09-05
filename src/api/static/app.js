// ─── Research Assistant Web App Logic ─────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  const queryForm = document.getElementById("queryForm");
  const queryInput = document.getElementById("queryInput");
  const topKInput = document.getElementById("topKInput");
  const maxIterInput = document.getElementById("maxIterInput");
  const submitBtn = document.getElementById("submitBtn");
  const healthBadge = document.getElementById("healthBadge");
  const healthText = document.getElementById("healthText");
  const emptyState = document.getElementById("emptyState");
  const timelineCard = document.getElementById("timelineCard");
  const timelineTrack = document.getElementById("timelineTrack");
  const answerCard = document.getElementById("answerCard");
  const answerText = document.getElementById("answerText");
  const groundedBadge = document.getElementById("groundedBadge");
  const confidenceBadge = document.getElementById("confidenceBadge");
  const latencyBadge = document.getElementById("latencyBadge");
  const citationsSection = document.getElementById("citationsSection");
  const citationsList = document.getElementById("citationsList");
  const rawTraceCode = document.getElementById("rawTraceCode");
  const copyBtn = document.getElementById("copyBtn");

  // Check backend health on startup
  checkHealth();
  setInterval(checkHealth, 30000);

  async function checkHealth() {
    try {
      const res = await fetch("/health");
      if (res.ok) {
        const data = await res.json();
        healthBadge.style.display = "flex";
        if (data.status === "healthy") {
          const gpuInfo = data.gpu_device_name ? ` (${data.gpu_device_name.split(' ')[0]})` : "";
          healthText.textContent = `System Ready${gpuInfo}`;
        } else {
          healthText.textContent = "Loading Models...";
        }
      }
    } catch (e) {
      healthText.textContent = "Offline";
    }
  }

  // Quick preset chips
  document.querySelectorAll(".chip").forEach(chip => {
    chip.addEventListener("click", () => {
      queryInput.value = chip.getAttribute("data-query");
      queryInput.focus();
    });
  });

  // Copy Answer
  copyBtn.addEventListener("click", () => {
    navigator.clipboard.writeText(answerText.innerText).then(() => {
      copyBtn.textContent = "Copied!";
      setTimeout(() => copyBtn.textContent = "Copy", 2000);
    });
  });

  // Form Submission via Server-Sent Events (SSE)
  queryForm.addEventListener("submit", async (e) => {
    e.preventDefault();

    const query = queryInput.value.trim();
    const top_k = parseInt(topKInput.value) || 10;
    const max_iterations = parseInt(maxIterInput.value) || 2;

    if (!query) return;

    // UI state: loading
    submitBtn.disabled = true;
    submitBtn.querySelector(".btn-text").textContent = "Synthesizing...";
    emptyState.style.display = "none";
    timelineCard.style.display = "block";
    answerCard.style.display = "none";
    timelineTrack.innerHTML = "";

    const startTime = Date.now();

    try {
      const response = await fetch("/api/v1/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, top_k, max_iterations }),
      });

      if (!response.ok) {
        throw new Error(`Server returned HTTP ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) {
          if (buffer && buffer.trim()) {
            handleSseBlock(buffer, Date.now() - startTime);
          }
          break;
        }

        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop(); // Keep incomplete chunk

        for (const block of parts) {
          if (!block.trim()) continue;
          handleSseBlock(block, Date.now() - startTime);
        }
      }
    } catch (err) {
      console.error(err);
      addTimelineStep("error", "❌ Pipeline Error", err.message || "Failed to connect to agent server.");
    } finally {
      submitBtn.disabled = false;
      submitBtn.querySelector(".btn-text").textContent = "Execute Multi-Agent Research";
    }
  });

  function handleSseBlock(block, elapsedMs) {
    let eventName = "message";
    let dataStr = "";

    const lines = block.split("\n");
    for (const line of lines) {
      if (line.startsWith("event:")) {
        eventName = line.replace("event:", "").trim();
      } else if (line.startsWith("data:")) {
        dataStr = line.replace("data:", "").trim();
      }
    }

    if (!dataStr) return;

    try {
      const payload = JSON.parse(dataStr);
      renderEventStep(eventName, payload, elapsedMs);
    } catch (e) {
      console.warn("Could not parse SSE JSON:", dataStr);
    }
  }

  function renderEventStep(event, payload, elapsedMs) {
    const step = payload.step || event;
    const msg = payload.message || "";
    const data = payload.data || {};

    if (step === "retrieval_start") {
      addTimelineStep("retrieval", "🔍 Step 1: Hybrid Retrieval", msg);
    } else if (step === "retrieval_done") {
      updateTimelineStep("retrieval", `Retrieved ${data.num_chunks} document chunks from FAISS + BM25.`);
    } else if (step === "reader_start") {
      addTimelineStep("reader", "📖 Step 2: Reader Agent", msg);
    } else if (step === "reader_done") {
      updateTimelineStep("reader", `Extracted ${data.num_passages} grounded evidence passages.`);
    } else if (step === "synthesizer_start") {
      addTimelineStep("synthesizer", `✍️ Step 3: Synthesizer Agent (Pass ${data.iteration})`, msg);
    } else if (step === "synthesizer_done") {
      updateTimelineStep("synthesizer", `Generated cited draft with ${data.citation_ids.length} citations.`);
    } else if (step === "critic_start") {
      addTimelineStep("critic", "⚖️ Step 4: Critic Agent Validation", msg);
    } else if (step === "critic_done") {
      const verdict = data.is_grounded ? "✅ Approved (Grounded)" : "⚠️ Refinement Needed";
      updateTimelineStep("critic", `${verdict} — Confidence: ${data.confidence.toFixed(2)}.`);
    } else if (step === "complete") {
      renderFinalAnswer(data, elapsedMs);
    } else if (step === "error") {
      addTimelineStep("error", "❌ Error", msg);
    }
  }

  function addTimelineStep(id, title, description) {
    const stepEl = document.createElement("div");
    stepEl.className = "timeline-step active";
    stepEl.id = `step-${id}`;
    stepEl.innerHTML = `
      <div class="step-icon">⏳</div>
      <div class="step-content">
        <h4>${title}</h4>
        <p>${description}</p>
      </div>
    `;
    timelineTrack.appendChild(stepEl);
  }

  function updateTimelineStep(id, description) {
    const stepEl = document.getElementById(`step-${id}`);
    if (stepEl) {
      stepEl.classList.remove("active");
      stepEl.querySelector(".step-icon").textContent = "✓";
      stepEl.querySelector(".step-icon").style.color = "#10b981";
      stepEl.querySelector("p").textContent = description;
    }
  }

  function renderFinalAnswer(data, elapsedMs) {
    // Unhide and display the answer card
    answerCard.style.display = "block";

    function escapeHtml(str) {
      if (!str) return "";
      return str
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
    }

    const evidenceList = data.evidence || [];
    const evidenceMap = {};
    evidenceList.forEach(e => { evidenceMap[e.id] = e; });

    // Format Evidence tags in answer body with clickable anchor links and preview tooltips
    let formattedAnswer = data.answer || "";
    // Safeguard: collapse any repeated consecutive duplicate evidence citations
    formattedAnswer = formattedAnswer.replace(/(\[Evidence\s*\d+\])(?:\s*,?\s*\1)+/gi, "$1");
    formattedAnswer = formattedAnswer.replace(/\[Evidence\s*(\d+)\]/g, (match, id) => {
      const ev = evidenceMap[id];
      const preview = ev ? escapeHtml(ev.text.substring(0, 140)) + "..." : "Evidence passage " + id;
      return `<a href="#evidence-${id}" class="evidence-tag" title="${preview}">[Evidence ${id}]</a>`;
    });
    answerText.innerHTML = formattedAnswer;

    // Badges
    const isGrounded = data.is_grounded !== false;
    groundedBadge.textContent = isGrounded ? "Grounded (No Hallucinations)" : "Hallucination Alert";
    groundedBadge.style.color = isGrounded ? "#10b981" : "#f59e0b";

    confidenceBadge.textContent = `Confidence: ${(data.confidence || 1.0).toFixed(2)}`;
    latencyBadge.textContent = `⏱️ ${(elapsedMs / 1000).toFixed(1)}s`;

    // Render Full Verified Citations & Passages
    citationsList.innerHTML = "";
    if (evidenceList.length > 0) {
      evidenceList.forEach((item) => {
        const cleanId = (item.arxiv_id || item.source || "").replace("arXiv:", "").trim();
        const card = document.createElement("div");
        card.className = "citation-card";
        card.id = `evidence-${item.id}`;
        card.innerHTML = `
          <div class="citation-card-header">
            <span class="evidence-pill">[Evidence ${item.id}]</span>
            <span class="evidence-paper">Paper: <strong>${escapeHtml(item.source)}</strong></span>
            <a href="https://arxiv.org/abs/${cleanId}" target="_blank" rel="noreferrer">View Paper on arXiv ↗</a>
          </div>
          <div class="citation-quote">
            <blockquote>"${escapeHtml(item.text)}"</blockquote>
          </div>
        `;
        citationsList.appendChild(card);
      });
      citationsSection.style.display = "block";
    } else if (data.sources && data.sources.length > 0) {
      data.sources.forEach((src, idx) => {
        const cleanId = src.replace("arXiv:", "").trim();
        const card = document.createElement("div");
        card.className = "citation-card";
        card.innerHTML = `
          <div class="citation-card-header">
            <span class="evidence-pill">[Source ${idx + 1}]</span>
            <span class="evidence-paper">${escapeHtml(src)}</span>
            <a href="https://arxiv.org/abs/${cleanId}" target="_blank" rel="noreferrer">View Paper ↗</a>
          </div>
        `;
        citationsList.appendChild(card);
      });
      citationsSection.style.display = "block";
    } else {
      citationsSection.style.display = "none";
    }

    // Populate Raw Trace
    rawTraceCode.textContent = JSON.stringify(data.reasoning_trace || [], null, 2);

    // Scroll smoothly to answer
    answerCard.scrollIntoView({ behavior: "smooth" });
  }
});
