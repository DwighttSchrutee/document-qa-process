import { createApp, ref, reactive, computed, onMounted, onUnmounted, watch } from "/vendor/vue.esm-browser.prod.js";

const api = async (url, opts = {}) => {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
};
const route = ref(location.hash.slice(1) || "/chat");
window.addEventListener("hashchange", () => (route.value = location.hash.slice(1) || "/chat"));
const go = (h) => (location.hash = h);
const unitKey = (u) => (u || []).join("/");

// ------------------------------------------------------------------ documents + upload
const Documents = {
  setup() {
    const docs = ref([]), jobs = ref([]), over = ref(false), msg = ref(""), picker = ref(null), uploading = ref(false);
    const load = async () => {
      docs.value = await api("/documents");
      jobs.value = await api("/jobs");
    };
    let t;
    onMounted(() => { load(); t = setInterval(load, 5000); });
    onUnmounted(() => clearInterval(t));
    const upload = async (files) => {
      uploading.value = true;
      for (const f of Array.from(files)) {
        msg.value = `${f.name}: uploading…`;
        const fd = new FormData();
        fd.append("file", f);
        try {
          const r = await api("/documents", { method: "POST", body: fd });
          msg.value = r.duplicate ? `${f.name}: identical to ${r.filename}, not ingested again` : `${f.name}: queued`;
        } catch (e) { msg.value = `${f.name}: ${e.message}`; }
      }
      uploading.value = false;
      load();
    };
    const drop = (e) => { over.value = false; upload(e.dataTransfer.files); };
    return { docs, jobs, over, msg, upload, drop, go, picker, uploading };
  },
  template: `
  <h2>Documents</h2>
  <div class="card drop" :class="{over}" @dragover.prevent="over=true" @dragleave="over=false" @drop.prevent="drop">
    Drop PDF, Word, Excel, CSV or text files here, or
    <button class="primary" type="button" :disabled="uploading" @click="picker.click()">{{ uploading ? 'Uploading…' : 'Choose files' }}</button>
    <input ref="picker" type="file" multiple accept=".pdf,.docx,.xlsx,.xlsm,.csv,.txt,.md" style="display: none"
           @change="upload($event.target.files); $event.target.value = ''">
    <div class="muted small" v-if="msg">{{ msg }}</div>
  </div>
  <div class="card" v-if="jobs.length">
    <b>Ingestion jobs</b>
    <table><tr><th>file</th><th>status</th><th>progress</th></tr>
      <tr v-for="j in jobs" :key="j.id"><td>{{ j.filename }}</td><td>{{ j.status }}<span v-if="j.error" class="muted"> — {{ j.error }}</span></td>
      <td class="small muted">{{ j.progress }}</td></tr></table>
  </div>
  <table>
    <tr><th>file</th><th>pages</th><th>quality report</th><th></th></tr>
    <tr v-for="d in docs" :key="d.sha256">
      <td>{{ d.filename }}</td><td>{{ d.pages || '' }}</td>
      <td><span v-if="d.ingesting" class="muted">ingesting…</span><span v-else-if="d.gate">{{ d.gate.summary }}</span></td>
      <td><a v-if="!d.ingesting" :href="'#/report/' + d.sha256">report</a></td>
    </tr>
  </table>`,
};

const Report = {
  props: ["sha"],
  setup(props) {
    const r = ref(null), err = ref("");
    onMounted(async () => { try { r.value = await api(`/documents/${props.sha}/report`); } catch (e) { err.value = e.message; } });
    return { r, err };
  },
  template: `
  <div v-if="err" class="card">{{ err }}</div>
  <div v-if="r">
    <h2>{{ r.filename }}</h2>
    <div class="card"><b>{{ r.gate && r.gate.summary }}</b>
      <div class="muted small">{{ r.seconds }} s for this run; duplicates are detected by SHA-256 ({{ r.sha256.slice(0, 12) }}…)</div></div>
    <table>
      <tr><th>page</th><th>status</th><th>method</th><th>corrections (numbers)</th><th>restored</th><th>unconfirmed</th><th>OCR</th><th>why</th></tr>
      <tr v-for="p in r.per_page" :key="p.page">
        <td><a :href="'#/review/' + r.sha256 + '/' + p.page">p{{ p.page }}</a></td>
        <td><span class="badge" :class="p.status">{{ p.status }}</span></td>
        <td>{{ p.method }}</td><td>{{ p.corrections }} ({{ p.corrections_num }})</td><td>{{ p.restored }}</td>
        <td>{{ p.unconfirmed }}</td><td class="small muted">{{ p.ocr_mode }} {{ p.ocr_confidence }}</td>
        <td class="small">{{ (p.reasons || []).map(x => x.detail).join('; ') }}</td>
      </tr>
    </table>
  </div>`,
};

// ------------------------------------------------------------------ review
const ReviewQueue = {
  setup() {
    const items = ref([]), warn = ref(false), loading = ref(true);
    const load = async () => {
      loading.value = true;
      items.value = await api(`/review/queue?include_warn=${warn.value}`);
      loading.value = false;
    };
    onMounted(load);
    watch(warn, load);
    return { items, warn, loading };
  },
  template: `
  <h2>Review queue</h2>
  <p class="muted small">HOLD items are not searchable until reviewed. WARN items are searchable, but answers citing them say they could not be verified.</p>
  <label class="small"><input type="checkbox" v-model="warn"> include warnings</label>
  <p v-if="loading" class="muted">loading…</p>
  <p v-else-if="!items.length" class="muted">Nothing to review.</p>
  <table v-else>
    <tr><th>status</th><th>document</th><th>where</th><th>check</th><th>value</th><th>text</th></tr>
    <tr v-for="(x, i) in items" :key="i">
      <td><span class="badge" :class="x.status">{{ x.status }}</span></td>
      <td>{{ x.filename }}</td>
      <td><a :href="'#/review/' + x.sha + '/' + x.page + '?u=' + x.unit.join('/')">p{{ x.page }} {{ x.unit[0] }} {{ x.unit.slice(1).join('.') }}</a></td>
      <td class="small">{{ x.check }}<div class="muted">{{ x.detail }}</div></td>
      <td><b>{{ x.value }}</b></td><td class="small">{{ x.text }}</td>
    </tr>
  </table>`,
};

const ReviewPage = {
  props: ["sha", "page", "sel"],
  setup(props) {
    const p = ref(null), err = ref(""), selected = ref(props.sel || ""), corr = reactive({}), busy = ref(false);
    const img = ref(null), imgW = ref(0);
    const load = async () => {
      try { p.value = await api(`/pages/${props.sha}/${props.page}`); } catch (e) { err.value = e.message; }
    };
    onMounted(load);
    const items = computed(() => (p.value && p.value.gate ? p.value.gate.items : []));
    const reasons = computed(() => (p.value && p.value.gate ? p.value.gate.reasons : []));
    const statusOf = (unit) => {
      const st = items.value.filter((x) => unitKey(x.unit) === unitKey(unit)).map((x) => x.status);
      return ["REJECTED", "HOLD", "WARN"].find((s) => st.includes(s)) || "";
    };
    const valuesOf = (unit) => items.value.filter((x) => unitKey(x.unit) === unitKey(unit) && x.value).map((x) => x.value);
    const esc = (s) => s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
    const marked = (text, unit) => {
      let h = esc(text || "");
      for (const v of valuesOf(unit)) h = h.split(esc(v)).join(`<mark>${esc(v)}</mark>`);
      return h;
    };
    const scale = computed(() => (p.value && p.value.size && imgW.value ? imgW.value / p.value.size[0] : 0));
    const boxStyle = (b) => {
      const s = scale.value;
      return b && s ? { left: b[0] * s + "px", top: b[1] * s + "px", width: (b[2] - b[0]) * s + "px", height: (b[3] - b[1]) * s + "px" } : { display: "none" };
    };
    const onImg = () => { imgW.value = img.value ? img.value.clientWidth : 0; };
    window.addEventListener("resize", onImg);
    const decide = async (x, decision) => {
      busy.value = true;
      const body = { unit: x.unit, decision, value: x.value };
      if (decision === "correct") body.corrected = corr[unitKey(x.unit) + x.value];
      try {
        const r = await api(`/review/${props.sha}/${props.page}`, {
          method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
        });
        p.value = r.page;
      } catch (e) { alert(e.message); }
      busy.value = false;
    };
    const decidePage = (decision) => decide({ unit: ["page"], value: null }, decision);
    return { p, err, items, reasons, statusOf, marked, boxStyle, img, onImg, selected, unitKey, corr, decide, decidePage, busy };
  },
  template: `
  <div v-if="err" class="card">{{ err }}</div>
  <div v-if="p">
    <div class="row" style="justify-content: space-between">
      <h2>Page {{ p.page_no }} <span class="badge" :class="p.gate && p.gate.status">{{ p.gate && p.gate.status }}</span>
        <span class="muted small"> method: {{ p.method }}</span></h2>
      <div class="row">
        <a :href="'#/review/' + sha + '/' + (p.page_no - 1)" v-if="p.page_no > 1">← previous</a>
        <a :href="'#/review/' + sha + '/' + (p.page_no + 1)">next →</a>
      </div>
    </div>
    <div class="card" v-if="reasons.length">
      <div v-for="r in reasons" class="row"><span class="badge" :class="r.status">{{ r.status }}</span> {{ r.check }}: {{ r.detail }}
        <span v-if="r.reviewed" class="muted small">(reviewed: {{ r.reviewed }})</span></div>
      <div class="row" style="margin-top: 8px" v-if="p.gate.status === 'HOLD'">
        <button class="primary" :disabled="busy" @click="decidePage('accept')">Accept page (index it)</button>
        <button :disabled="busy" @click="decidePage('reject')">Reject page</button>
      </div>
    </div>
    <div class="split">
      <div class="sticky">
        <div class="pageimg" v-if="p.size">
          <img ref="img" :src="'/pages/' + sha + '/' + p.page_no + '/image?dpi=110'" @load="onImg">
          <div v-for="(x, i) in items" :key="i" class="box" :class="[x.status, {sel: unitKey(x.unit) === selected}]" :style="boxStyle(x.bbox)"></div>
        </div>
        <div v-else class="muted">No page image for this file type.</div>
      </div>
      <div>
        <div class="card" v-if="items.length">
          <b>Flagged values</b>
          <ul class="items">
            <li v-for="(x, i) in items" :key="i" @mouseenter="selected = unitKey(x.unit)">
              <span class="badge" :class="x.status">{{ x.status }}</span>
              <b> {{ x.value }}</b> <span class="small muted">{{ x.check }} — {{ x.detail }} ({{ x.unit.join(' ') }})</span>
              <span v-if="x.reviewed" class="small muted"> reviewed: {{ x.reviewed }}</span>
              <div class="row small" v-if="x.status === 'HOLD' || x.status === 'WARN'">
                <button :disabled="busy" @click="decide(x, 'accept')">accept</button>
                <input type="text" placeholder="correct value" v-model="corr[unitKey(x.unit) + x.value]" size="14" v-if="x.value">
                <button :disabled="busy || !corr[unitKey(x.unit) + x.value]" @click="decide(x, 'correct')" v-if="x.value">correct</button>
                <button :disabled="busy" @click="decide(x, 'reject')">reject row</button>
              </div>
            </li>
          </ul>
        </div>
        <div v-for="b in p.blocks" :key="b.id" class="block">
          <div class="label">{{ b.label }}</div>
          <div v-if="b.kind === 'table'" class="tbl"><table>
            <tr v-for="(r, i) in b.table.rows" :key="i" :class="statusOf(['row', b.id, i])">
              <td v-for="(c, j) in r" :key="j" v-html="marked(c, ['row', b.id, i])"></td></tr></table></div>
          <div v-else-if="b.kind === 'text'">
            <div v-for="(ln, i) in b.lines" :key="i" class="unit" :class="[statusOf(['line', b.id, i]), {sel: unitKey(['line', b.id, i]) === selected}]"
                 v-html="marked(ln, ['line', b.id, i])"></div></div>
        </div>
      </div>
    </div>
  </div>`,
};

// ------------------------------------------------------------------ chat
const escHtml = (s) => (s || "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
function renderAnswer(md) {
  // small markdown subset: tables, bullet lists, bold; [n] becomes a clickable citation
  const lines = escHtml(md).split("\n");
  let html = "", i = 0;
  while (i < lines.length) {
    if (/^\s*\|.*\|\s*$/.test(lines[i])) {
      const rows = [];
      while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) rows.push(lines[i++]);
      const cells = (r) => r.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
      const body = rows.filter((r) => !/^\s*\|[\s:\-|]+\|\s*$/.test(r));
      html += "<table class='ans'>" + body.map((r, k) => "<tr>" + cells(r).map((c) => (k ? "<td>" : "<th>") + c + (k ? "</td>" : "</th>")).join("") + "</tr>").join("") + "</table>";
      continue;
    }
    if (/^\s*[-*]\s+/.test(lines[i])) {
      html += "<ul>";
      while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) html += "<li>" + lines[i++].replace(/^\s*[-*]\s+/, "") + "</li>";
      html += "</ul>";
      continue;
    }
    html += lines[i].trim() ? "<p>" + lines[i] + "</p>" : "";
    i++;
  }
  return html.replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/\[([0-9০-৯]{1,2})\]/g, (m, n) => {
    const k = n.replace(/[০-৯]/g, (d) => "০১২৩৪৫৬৭৮৯".indexOf(d));     // Bangla digits in citations
    return `<a class='cite' data-n='${k}'>${m}</a>`;
  });
}

const Chat = {
  setup() {
    const q = ref(""), think = ref(false), turns = ref([]), busy = ref(false), open = ref(null);
    const ask = async () => {
      const question = q.value.trim();
      if (!question || busy.value) return;
      q.value = "";
      busy.value = true;
      const t = reactive({ question, answer: "", thinking: "", sources: [], status: "", checks: null, done: null, round: 1, notes: [],
        addOpen: false, add: { answerable: true, expected: "", must: "", gold: [], msg: "" } });
      turns.value.push(t);
      const history = turns.value.slice(0, -1).filter((x) => x.done).flatMap((x) => [
        { role: "user", content: x.question }, { role: "assistant", content: x.done.answer }]);
      try {
        const r = await fetch("/ask", { method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question, think: think.value, history }) });
        const reader = r.body.getReader(), dec = new TextDecoder();
        let buf = "";
        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          buf += dec.decode(value, { stream: true });
          let k;
          while ((k = buf.indexOf("\n")) >= 0) {
            const line = buf.slice(0, k); buf = buf.slice(k + 1);
            if (!line.trim()) continue;
            const ev = JSON.parse(line);
            if (ev.type === "token") t.answer += ev.text;
            else if (ev.type === "thinking") t.thinking += ev.text;
            else if (ev.type === "status") t.status = ev.text;
            else if (ev.type === "sources") { t.sources = ev.passages; t.round = ev.round; }
            else if (ev.type === "reset") { t.answer = ""; t.notes.push(ev.reason); }
            else if (ev.type === "checks") t.checks = ev;
            else if (ev.type === "done") { t.done = ev; t.answer = ev.answer; t.status = ""; }
            else if (ev.type === "error") { t.status = ""; t.answer += "\n\nError: " + ev.text; }
          }
        }
      } catch (e) { t.answer += "\n\nError: " + e.message; }
      busy.value = false;
    };
    const click = (t, e) => {
      const a = e.target.closest("a.cite");
      if (!a) return;
      const s = t.sources[+a.dataset.n - 1];
      if (s) open.value = s;
    };
    const cropUrl = (s) => {
      const page = (s.pages || [])[0];
      const b = s.bbox && s.bbox[String(page)];
      return b ? `/pages/${s.sha256}/${page}/crop?bbox=${b.join(",")}&dpi=150` : `/pages/${s.sha256}/${page}/image?dpi=90`;
    };
    const fmt = (x) => (x == null ? "" : x);
    const saveTest = async (t) => {
      const gold = t.add.gold.flatMap((n) => { const s = t.sources[n - 1]; return (s.pages || []).map((p) => ({ file: s.filename, page: p })); });
      try {
        const r = await api("/eval/questions", { method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: t.question, expected_answer: t.add.answerable ? t.add.expected : "not in the documents",
            must_contain: t.add.answerable ? t.add.must.split(",").map((x) => x.trim()).filter(Boolean) : [],
            answerable: t.add.answerable, gold_sources: t.add.answerable ? gold : [] }) });
        t.add.msg = `saved (${r.count} questions in ${r.file})`;
      } catch (e) { t.add.msg = e.message; }
    };
    return { q, think, turns, busy, ask, open, click, renderAnswer, cropUrl, fmt, saveTest };
  },
  template: `
  <div class="chat">
    <div v-if="!turns.length" class="card muted">
      Ask about the uploaded documents in Bangla or English. Every statement is cited; click [n] to see the passage and
      the page region it came from. Examples: <i>What is the minimum value for shampoos (HS 3305.10.00)?</i> ·
      <i>সাধারণ বন্ড কত বছরের জন্য জারি করা যায়?</i>
    </div>
    <div class="turn card" v-for="(t, i) in turns" :key="i">
      <div class="question">{{ t.question }}</div>
      <details v-if="t.thinking" class="small muted"><summary>reasoning</summary><pre class="think">{{ t.thinking }}</pre></details>
      <div class="muted small" v-if="t.status">{{ t.status }}…</div>
      <div class="answer" :class="{notfound: t.done && t.done.not_found}" v-html="renderAnswer(t.answer.replace('[NOT FOUND]', ''))" @click="click(t, $event)"></div>
      <div v-if="t.done && t.done.not_found" class="card small">
        <b>Not found in the documents.</b> Closest passages:
        <ul><li v-for="c in t.done.closest || []"><a class="cite" @click="open = c">{{ c.filename }} p{{ (c.pages || []).join(',') }}</a> — {{ (c.text || '').slice(0, 140) }}…</li></ul>
      </div>
      <div v-for="w in (t.done && t.done.warnings) || []" class="warnline">⚠ {{ w }}</div>
      <div class="sources small" v-if="t.sources.length">
        <span class="muted">Sources{{ t.round > 1 ? ' (second search)' : '' }}:</span>
        <a v-for="s in t.sources" :key="s.n" class="src" :class="s.status" @click="open = s">[{{ s.n }}] {{ s.filename }} p{{ (s.pages || []).join(',') }}<span v-if="s.status === 'WARN'"> ⚠</span></a>
      </div>
      <div class="muted small" v-if="t.done">{{ t.done.timings.total }} s
        <span v-if="t.notes.length"> · retried: {{ t.notes.join('; ') }}</span>
        · <a href="javascript:void 0" @click="t.addOpen = !t.addOpen">add to test set</a></div>
      <div v-if="t.addOpen" class="card small addtest">
        <div class="row"><label><input type="radio" :value="true" v-model="t.add.answerable"> answerable</label>
          <label><input type="radio" :value="false" v-model="t.add.answerable"> not in the documents</label></div>
        <template v-if="t.add.answerable">
          <input type="text" v-model="t.add.expected" placeholder="correct answer (short)" style="width: 100%">
          <input type="text" v-model="t.add.must" placeholder="values the answer must contain, comma-separated, e.g. 4.00, kg" style="width: 100%">
          <div>gold pages: <label v-for="s in t.sources" :key="s.n" style="margin-right: 10px">
            <input type="checkbox" :value="s.n" v-model="t.add.gold"> [{{ s.n }}]</label></div>
        </template>
        <div class="row"><button class="primary" @click="saveTest(t)">save</button><span class="muted">{{ t.add.msg }}</span></div>
      </div>
    </div>
    <div class="askbar">
      <textarea v-model="q" rows="2" placeholder="Ask in Bangla or English…  (Enter to send, Shift+Enter for a new line)"
        @keydown.enter.exact.prevent="ask"></textarea>
      <div class="row"><button class="primary" :disabled="busy || !q.trim()" @click="ask">{{ busy ? 'Answering…' : 'Ask' }}</button>
        <label class="small"><input type="checkbox" v-model="think"> reasoning (slower)</label></div>
    </div>
    <div class="drawer" v-if="open" @click.self="open = null">
      <div class="drawer-body card">
        <div class="row" style="justify-content: space-between"><b>{{ open.filename }} — page {{ (open.pages || []).join(', ') }}</b>
          <button @click="open = null">close</button></div>
        <div class="small muted">{{ (open.section_path || []).join(' / ') }}<span v-if="open.clause_no"> · clause {{ open.clause_no }}</span>
          <span v-if="open.issued"> · issued {{ open.issued }}</span><span v-if="open.effective_from"> · effective {{ open.effective_from }} – {{ fmt(open.effective_to) || 'open' }}</span>
          <span v-if="open.added_for"> · this notice {{ open.added_for }}</span></div>
        <div v-if="open.status === 'WARN'" class="warnline">⚠ Could not be verified against the document (quality gate warning).</div>
        <img v-if="open.sha256 && open.pages && open.pages.length" :src="cropUrl(open)" class="crop">
        <pre class="passage">{{ open.text }}</pre>
        <a v-if="open.sha256" :href="'#/review/' + open.sha256 + '/' + open.pages[0]">open the full page</a>
      </div>
    </div>
  </div>`,
};

// ------------------------------------------------------------------ evaluation dashboard
const Eval = {
  setup() {
    const runs = ref([]), sel = ref(null), err = ref("");
    onMounted(async () => {
      try { runs.value = await api("/eval/latest"); if (runs.value.length) sel.value = await api(`/eval/run/${runs.value[0].id}`); }
      catch (e) { err.value = e.message; }
    });
    const pick = async (id) => { sel.value = await api(`/eval/run/${id}`); };
    const pct = (x) => (x == null ? "–" : (100 * x).toFixed(1) + "%");
    return { runs, sel, err, pick, pct };
  },
  template: `
  <h2>Evaluation</h2>
  <div v-if="err" class="card">{{ err }}</div>
  <p v-if="!runs.length" class="muted">No evaluation run yet. Add questions to eval/questions_*.jsonl and run <code>make eval</code>.</p>
  <div class="row small" v-if="runs.length">Runs:
    <a v-for="r in runs" :key="r.id" @click="pick(r.id)" href="javascript:void 0">#{{ r.id }} {{ r.name }} ({{ r.at.slice(0, 16) }})</a></div>
  <div v-if="sel" class="card">
    <b>Run #{{ sel.id }} — {{ sel.name }}</b>
    <table><tr><th>group</th><th>n</th><th>answer accuracy</th><th>recall@8</th><th>citation accuracy</th><th>false not-found</th><th>hallucinated numbers</th><th>p50 / p95 s</th></tr>
      <tr v-for="(m, g) in sel.metrics.groups" :key="g"><td>{{ g }}</td><td>{{ m.n }}</td><td>{{ pct(m.accuracy) }}</td><td>{{ pct(m.recall8) }}</td>
        <td>{{ pct(m.citation_accuracy) }}</td><td>{{ pct(m.false_not_found) }}</td><td>{{ m.hallucinated_numbers }}</td><td>{{ m.p50 }} / {{ m.p95 }}</td></tr></table>
    <details><summary>failures</summary>
      <div v-for="r in sel.results.filter(x => !x.correct)" :key="r.id" class="small" style="margin: 8px 0">
        <b>{{ r.id }}</b> [{{ r.category }}] {{ r.question }}<br><span class="muted">expected:</span> {{ r.expected }}<br>
        <span class="muted">answer:</span> {{ (r.answer || '').slice(0, 400) }}<br><span class="muted">why:</span> {{ (r.why || []).join('; ') }}</div>
    </details>
  </div>`,
};

// ------------------------------------------------------------------ shell
const App = {
  components: { Documents, Report, ReviewQueue, ReviewPage, Chat, Eval },
  setup() {
    const view = computed(() => {
      const [path, query] = route.value.split("?");
      const parts = path.split("/").filter(Boolean);
      const q = new URLSearchParams(query || "");
      if (parts[0] === "report") return { c: "Report", p: { sha: parts[1] } };
      if (parts[0] === "review" && parts[1]) return { c: "ReviewPage", p: { sha: parts[1], page: +parts[2], sel: q.get("u") || "" } };
      if (parts[0] === "review") return { c: "ReviewQueue", p: {} };
      if (parts[0] === "documents") return { c: "Documents", p: {} };
      if (parts[0] === "eval") return { c: "Eval", p: {} };
      return { c: "Chat", p: {} };
    });
    const tab = computed(() => (route.value.split("/")[1] || "chat").split("?")[0]);
    return { view, tab };
  },
  template: `
  <header>
    <b>Document Q&amp;A</b>
    <a href="#/chat" :class="{on: tab === 'chat'}">Ask</a>
    <a href="#/documents" :class="{on: tab === 'documents' || tab === 'report'}">Documents</a>
    <a href="#/review" :class="{on: tab === 'review'}">Review</a>
    <a href="#/eval" :class="{on: tab === 'eval'}">Evaluation</a>
    <span class="muted small" style="margin-left:auto">runs locally — nothing leaves this computer</span>
  </header>
  <main><component :is="view.c" v-bind="view.p" :key="JSON.stringify(view)"></component></main>`,
};
createApp(App).mount("#app");
