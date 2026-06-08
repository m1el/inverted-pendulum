#!/usr/bin/env python3
"""Collect Claude Code + Codex sessions for /var/data/bootstrap into ./sessions."""
import json, os, glob, re, getpass, subprocess, urllib.request
from datetime import datetime

PROJECT = "/var/data/inverted-pendulum"
CLAUDE_DIR = os.path.expanduser("~/.claude/projects/" + PROJECT.replace("/", "-"))
CODEX_GLOB = os.path.expanduser("~/.codex/sessions/**/*.jsonl")
OUT = os.path.join(PROJECT, "sessions")
TOOL_TRUNC = 4000   # max chars per tool-result/output block in the transcript

# Redaction: applied to every byte written under sessions/ (transcripts + README).
# Targets are DISCOVERED at runtime (git identity + the OS user/home) so that no
# personal data is hard-coded — this script can itself be published next to the
# output without leaking what it redacts. Matched as literal substrings (no \b
# anchors) so even values embedded in quoted source/JSON inside a transcript are
# caught. Username is only redacted in home-path context (a bare "null" would
# clobber JSON `null`).
def _git_cfg(key):
    try:
        return subprocess.check_output(["git", "-C", PROJECT, "config", key],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""

def build_redactions():
    reds = []
    email = _git_cfg("user.email")
    name = _git_cfg("user.name")
    home = os.path.expanduser("~")
    try:
        user = getpass.getuser()
    except Exception:
        user = ""
    if email:
        reds.append((re.compile(re.escape(email)), "<redacted-email>"))
    if name:
        reds.append((re.compile(re.escape(name)), "<redacted-name>"))
    if home and home not in ("~", "/"):
        reds.append((re.compile(re.escape(home)), "/home/<user>"))
    elif user:
        reds.append((re.compile(re.escape(f"/home/{user}")), "/home/<user>"))
    return reds

REDACTIONS = build_redactions()

def redact(s):
    if not isinstance(s, str):
        return s
    for pat, repl in REDACTIONS:
        s = pat.sub(repl, s)
    return s

def jlines(path):
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                yield json.loads(ln)
            except Exception:
                continue

def trunc(s, n=TOOL_TRUNC):
    s = s if isinstance(s, str) else json.dumps(s, ensure_ascii=False)
    if len(s) > n:
        return s[:n] + f"\n… [truncated {len(s)-n} chars]"
    return s

def fmt_ts(ts):
    if not ts:
        return "?"
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M:%SZ")
    except Exception:
        return ts

def t2f(ts):
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None

def clock(ts):
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).strftime("%H:%M:%S")
    except Exception:
        return ts or ""

def fmt_dur(s):
    if s is None:
        return "?"
    s = float(s)
    if s < 1:
        return f"{s*1000:.0f}ms"
    if s < 60:
        return f"{s:.1f}s"
    m, sec = divmod(int(round(s)), 60)
    if m < 60:
        return f"{m}m{sec:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"

def render_sections(sections):
    """Turn timed sections into transcript markdown + a timing breakdown.

    Each section's interval [prev.end, this.end] is attributed to one phase by the
    section's kind: user→waiting-for-user, assistant→model, toolresult→tool. The
    intervals tile the whole session, so the buckets sum to wall-clock. Per-tool
    exec times (matched call→result) are rendered inline on the call lines."""
    out, timing = [], dict(waiting=0.0, model=0.0, tool=0.0, wall=0.0)
    prev_end = None
    for s in sections:
        ts = s.get("ts")
        end = s.get("end") if s.get("end") is not None else ts
        dur = (end - prev_end) if (prev_end is not None and end is not None) else None
        if dur is not None and dur >= 0:
            timing[{"user": "waiting", "assistant": "model", "toolresult": "tool"}[s["kind"]]] += dur
        if s["kind"] == "user":
            head, phase = "### 👤 **User**", "waited"
        elif s["kind"] == "assistant":
            head, phase = "### 🤖 **Assistant**", "model"
        else:
            head, phase = f"### 🛠️ **{s.get('hdr','Tool result')}**", "tool"
        meta = clock(s.get("ts_str", ""))
        if dur is not None and dur > 0:
            meta += f"; {phase} {fmt_dur(dur)}"
        out.append(f"{head}  ·  {meta}\n\n{s['body']}\n")
        if end is not None:
            prev_end = end
    if sections:
        f0 = sections[0].get("ts"); ln = sections[-1].get("end") or sections[-1].get("ts")
        if f0 is not None and ln is not None:
            timing["wall"] = ln - f0
    return "\n".join(out), timing

def first_prompt(text_blocks):
    """First human prompt that isn't a tag wrapper / caveat / system reminder."""
    for t in text_blocks:
        s = t.strip()
        if not s:
            continue
        # skip XML-ish wrappers (system-reminder, command-*, local-command-caveat,
        # environment_context, …) and command caveats / interrupt markers
        if s[0] == "<" or s.startswith("Caveat:") or s.startswith("[Request interrupted"):
            continue
        return s
    return text_blocks[0].strip() if text_blocks else ""

# ----------------------------------------------------------------- Claude
def parse_claude(path):
    sid = os.path.basename(path).replace(".jsonl", "")
    models, n_tool = set(), 0
    tok = dict(input=0, output=0, cache_create=0, cache_read=0)
    usage_by_req = {}
    user_prompts, last_assistant_text = [], ""
    recs = []   # (tf, ts_str, role, blocks, rid, model, usage) in order
    for d in jlines(path):
        if d.get("type") not in ("user", "assistant"):
            continue
        m = d.get("message", {}) or {}
        ts = d.get("timestamp")
        recs.append((t2f(ts), ts, m.get("role", d["type"]), m.get("content"),
                     d.get("requestId") or d.get("uuid"), m.get("model"), m.get("usage")))
    # match tool_use id -> tool_result for exact per-call exec times
    tool_open, tool_dur, tool_stats = {}, {}, {}
    for tf, ts, role, content, rid, model, usage in recs:
        for b in (content if isinstance(content, list) else []):
            if not isinstance(b, dict):
                continue
            if b.get("type") == "tool_use":
                tool_open[b.get("id")] = (tf, b.get("name"))
            elif b.get("type") == "tool_result":
                st = tool_open.get(b.get("tool_use_id"))
                if st and st[0] is not None and tf is not None:
                    dur = max(0.0, tf - st[0])
                    tool_dur[b["tool_use_id"]] = dur
                    tool_stats.setdefault(st[1] or "?", []).append(dur)
    # build timed sections
    sections, cur = [], None

    def flush():
        nonlocal cur
        if cur and cur["body"]:
            sections.append(dict(kind="assistant", ts=cur["ts"], end=cur["end"],
                                 ts_str=cur["ts_str"], body="\n\n".join(cur["body"])))
        cur = None

    for tf, ts, role, content, rid, model, usage in recs:
        blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]
        if role == "assistant":
            if model:
                models.add(model)
            if usage and rid not in usage_by_req:
                usage_by_req[rid] = usage
            rendered, texts = [], []
            for b in blocks:
                if not isinstance(b, dict):
                    continue
                bt = b.get("type")
                if bt == "text":
                    txt = b.get("text") or ""; texts.append(txt)
                    if txt.strip():
                        rendered.append(txt)
                elif bt == "thinking":
                    rendered.append("> 💭 (thinking)\n> " + trunc(b.get("thinking",""),1500).replace("\n","\n> "))
                elif bt == "tool_use":
                    n_tool += 1
                    dur = tool_dur.get(b.get("id"))
                    dtag = f"  ·  {fmt_dur(dur)}" if dur is not None else ""
                    rendered.append(f"**🔧 tool → {b.get('name')}**{dtag}\n```json\n{trunc(b.get('input',{}),1500)}\n```")
            if "\n".join(texts).strip():
                last_assistant_text = "\n".join(texts).strip()
            if cur is None or cur["rid"] != rid:
                flush(); cur = dict(rid=rid, ts=tf, end=tf, ts_str=ts, body=[])
            if tf is not None:
                cur["end"] = tf
            cur["body"].extend(r for r in rendered if r.strip())
        else:
            # a "user"-role message is either a human turn or the tool_result(s)
            # for the assistant's last calls — split them into distinct sections.
            flush()
            human, results = [], []
            for b in blocks:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "text" and (b.get("text") or "").strip():
                    human.append(b["text"])
                elif b.get("type") == "tool_result":
                    c = b.get("content")
                    if isinstance(c, list):
                        c = "\n".join(x.get("text","") if isinstance(x,dict) else str(x) for x in c)
                    tag = " (error)" if b.get("is_error") else ""
                    results.append(f"```\n{trunc(c)}\n```{tag}")
            if human:
                user_prompts.extend(human)
                sections.append(dict(kind="user", ts=tf, end=tf, ts_str=ts, body="\n\n".join(human)))
            if results:
                hdr = "Tool result" + ("s" if len(results) > 1 else "")
                sections.append(dict(kind="toolresult", ts=tf, end=tf, ts_str=ts,
                                     hdr=hdr, body="\n\n".join(results)))
    flush()
    for u in usage_by_req.values():
        tok["input"] += u.get("input_tokens", 0) or 0
        tok["output"] += u.get("output_tokens", 0) or 0
        tok["cache_create"] += u.get("cache_creation_input_tokens", 0) or 0
        tok["cache_read"] += u.get("cache_read_input_tokens", 0) or 0
    return dict(agent="claude", sid=sid, path=path, models=sorted(models),
                n_assist=len(usage_by_req),
                n_user=sum(1 for s in sections if s["kind"] == "user"),
                n_tool=n_tool, tok=tok, sections=sections, tool_stats=tool_stats,
                coarse_timing=False,
                first_ts=recs[0][1] if recs else None,
                last_ts=recs[-1][1] if recs else None,
                title=first_prompt(user_prompts), first_prompt=first_prompt(user_prompts),
                last_text=last_assistant_text)

# ----------------------------------------------------------------- Codex
def parse_codex(path):
    sid = cwd = model = None
    tok_info = None
    user_prompts, last_assistant_text = [], ""
    recs = []   # (tf, ts_str, ev-tuple) in order
    for d in jlines(path):
        typ = d.get("type"); p = d.get("payload", {}) or {}; ts = d.get("timestamp")
        if typ == "session_meta":
            sid = p.get("id"); cwd = p.get("cwd")
        elif typ == "turn_context":
            model = model or p.get("model"); cwd = cwd or p.get("cwd")
        elif typ == "event_msg" and p.get("type") == "token_count":
            info = p.get("info")
            if info and info.get("total_token_usage"):
                tok_info = info["total_token_usage"]
        elif typ == "event_msg" and p.get("type") == "user_message":
            m = p.get("message")
            if isinstance(m, str) and m.strip():
                recs.append((t2f(ts), ts, ("user", m)))
        elif typ == "response_item":
            pt = p.get("type")
            if pt == "message":
                role = p.get("role", "?")
                txt = "".join(c.get("text","") for c in (p.get("content") or []) if isinstance(c, dict))
                if not txt.strip():
                    continue
                recs.append((t2f(ts), ts, ("user" if role == "user" else "amsg", txt)))
            elif pt == "reasoning":
                summ = p.get("summary") or []
                rt = "\n".join(x.get("text","") if isinstance(x,dict) else str(x) for x in summ) if isinstance(summ,list) else str(summ)
                if rt.strip():
                    recs.append((t2f(ts), ts, ("reason", rt)))
            elif pt == "function_call":
                recs.append((t2f(ts), ts, ("call", p.get("name"), p.get("arguments",""), p.get("call_id"))))
            elif pt == "function_call_output":
                out = p.get("output")
                if isinstance(out, dict):
                    out = out.get("content", out)
                recs.append((t2f(ts), ts, ("out", out, p.get("call_id"))))
    # match call_id -> output for exec times
    call_open, tool_dur, tool_stats = {}, {}, {}
    for tf, ts, ev in recs:
        if ev[0] == "call":
            call_open[ev[3]] = (tf, ev[1])
        elif ev[0] == "out":
            st = call_open.get(ev[2])
            if st and st[0] is not None and tf is not None:
                dur = max(0.0, tf - st[0])
                tool_dur[ev[2]] = dur
                tool_stats.setdefault(st[1] or "?", []).append(dur)
    # build timed sections (group consecutive assistant-side events)
    sections, cur, n_tool = [], None, 0

    def flush():
        nonlocal cur
        if cur and cur["body"]:
            sections.append(dict(kind="assistant", ts=cur["ts"], end=cur["end"],
                                 ts_str=cur["ts_str"], body="\n\n".join(cur["body"])))
        cur = None

    for tf, ts, ev in recs:
        kind = ev[0]
        if kind in ("amsg", "reason", "call"):
            if cur is None:
                cur = dict(ts=tf, end=tf, ts_str=ts, body=[])
            if tf is not None:
                cur["end"] = tf
            if kind == "amsg":
                cur["body"].append(ev[1]); last_assistant_text = ev[1].strip()
            elif kind == "reason":
                cur["body"].append("> 💭 (reasoning)\n> " + trunc(ev[1],1500).replace("\n","\n> "))
            else:
                n_tool += 1
                dur = tool_dur.get(ev[3])
                dtag = f"  ·  {fmt_dur(dur)}" if dur is not None else ""
                cur["body"].append(f"**🔧 call → {ev[1]}**{dtag}\n```\n{trunc(ev[2],1500)}\n```")
        elif kind == "out":
            flush()
            sections.append(dict(kind="toolresult", ts=tf, end=tf, ts_str=ts,
                                 hdr="Tool output", body=f"```\n{trunc(ev[1])}\n```"))
        elif kind == "user":
            flush()
            user_prompts.append(ev[1])
            sections.append(dict(kind="user", ts=tf, end=tf, ts_str=ts, body=ev[1]))
    flush()
    tok = dict(input=0, output=0, cache_create=0, cache_read=0, reasoning=0, total=0)
    if tok_info:
        tok["input"] = tok_info.get("input_tokens", 0)
        tok["cache_read"] = tok_info.get("cached_input_tokens", 0)
        tok["output"] = tok_info.get("output_tokens", 0)
        tok["reasoning"] = tok_info.get("reasoning_output_tokens", 0)
        tok["total"] = tok_info.get("total_tokens", 0)
    return dict(agent="codex", sid=sid or os.path.basename(path), path=path,
                models=[model] if model else [], cwd=cwd,
                n_assist=sum(1 for s in sections if s["kind"] == "assistant"),
                n_user=sum(1 for s in sections if s["kind"] == "user"),
                n_tool=n_tool, tok=tok, sections=sections, tool_stats=tool_stats,
                coarse_timing=True,   # codex events are batch-flushed; timings approximate
                first_ts=recs[0][1] if recs else None,
                last_ts=recs[-1][1] if recs else None,
                title=first_prompt(user_prompts), first_prompt=first_prompt(user_prompts),
                last_text=last_assistant_text)

# ----------------------------------------------------------------- write
# ----------------------------------------------------------------- pricing
OPENROUTER_URL = "https://openrouter.ai/api/v1/models"
PRICE_CACHE = os.path.join(OUT, "openrouter_models.json")
PRICING = {}   # {"exact": {id: rates}, "base": {canon: (id, rates)}, "src": str}

def _canon(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())

def load_pricing():
    """Fetch OpenRouter model pricing (USD/token); cache to disk for offline reruns."""
    raw = src = None
    try:
        req = urllib.request.Request(OPENROUTER_URL, headers={"User-Agent": "sessions-collector"})
        raw = json.load(urllib.request.urlopen(req, timeout=30))
        os.makedirs(OUT, exist_ok=True)
        json.dump(raw, open(PRICE_CACHE, "w"))
        src = "openrouter.ai/api/v1/models (live)"
    except Exception as e:
        if os.path.exists(PRICE_CACHE):
            raw = json.load(open(PRICE_CACHE)); src = f"{PRICE_CACHE} (cache; live fetch failed: {e})"
        else:
            print("WARN: pricing unavailable (fetch failed, no cache):", e)
            return {"exact": {}, "base": {}, "src": None}
    exact, base = {}, {}
    for m in raw.get("data", []):
        mid = m["id"].lstrip("~")
        p = m.get("pricing", {}) or {}
        def g(k):
            try: return float(p[k])
            except Exception: return None
        rates = dict(prompt=g("prompt"), completion=g("completion"),
                     cache_read=g("input_cache_read"), cache_write=g("input_cache_write"))
        exact[mid.lower()] = rates
        base.setdefault(_canon(mid.split("/")[-1]), (mid, rates))
    return {"exact": exact, "base": base, "src": src}

def match_pricing(model):
    """Map a recorded model name to an OpenRouter id + rates. Returns (id, rates) or (None, None)."""
    if not PRICING or not model:
        return None, None
    ml = model.lower()
    if ml in PRICING["exact"]:
        return model, PRICING["exact"][ml]
    cb = _canon(model.split("/")[-1])
    if cb in PRICING["base"]:
        return PRICING["base"][cb]
    return None, None

def session_cost(sess):
    """USD cost from matched OpenRouter rates, or None if no pricing matched."""
    rates = sess.get("priced")
    if not rates:
        return None
    t = sess["tok"]
    prm = rates["prompt"] or 0.0
    cmp = rates["completion"] or 0.0
    cr  = rates["cache_read"]  if rates["cache_read"]  is not None else prm
    cw  = rates["cache_write"] if rates["cache_write"] is not None else prm
    if sess["agent"] == "claude":
        return t["input"]*prm + t["cache_create"]*cw + t["cache_read"]*cr + t["output"]*cmp
    # codex: input_tokens already includes cached; output_tokens already includes reasoning
    fresh = max(0, t["input"] - t["cache_read"])
    return fresh*prm + t["cache_read"]*cr + t["output"]*cmp

def file_stamp(sess):
    """yyyy-mm-dd_hh-mm from the session start (fallback: source-file mtime)."""
    ts = sess.get("first_ts")
    if ts:
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00")).strftime("%Y-%m-%d_%H-%M")
        except Exception:
            pass
    try:
        return datetime.fromtimestamp(os.path.getmtime(sess["path"])).strftime("%Y-%m-%d_%H-%M")
    except Exception:
        return "0000-00-00_00-00"

def timing_block(sess):
    tm = sess.get("timing") or {}
    wall, model, tool, wait = tm.get("wall",0), tm.get("model",0), tm.get("tool",0), tm.get("waiting",0)
    note = "  _(codex events are batch-flushed → timings approximate)_" if sess.get("coarse_timing") else ""
    out = [f"- wall-clock: **{fmt_dur(wall)}**  ·  active (model+tool): **{fmt_dur(model+tool)}**  ·  "
           f"waiting for user: {fmt_dur(wait)}{note}",
           f"- model generation: {fmt_dur(model)}  ·  tool execution: {fmt_dur(tool)}"]
    stats = sess.get("tool_stats") or {}
    if stats:
        top = sorted(stats.items(), key=lambda x: -sum(x[1]))[:6]
        out.append("- tool time by name: " + ", ".join(f"`{n}` {fmt_dur(sum(v))}/{len(v)}" for n, v in top))
        slow = sorted(((d, n) for n, v in stats.items() for d in v), reverse=True)[:3]
        if slow and slow[0][0] >= 1:
            out.append("- slowest calls: " + ", ".join(f"`{n}` {fmt_dur(d)}" for d, n in slow))
    return "## Timing\n\n" + "\n".join(out) + "\n\n"

def write_session(sess, subdir):
    d = os.path.join(OUT, subdir)
    os.makedirs(d, exist_ok=True)
    hexid = re.sub(r"[^0-9a-f]", "", sess["sid"].lower())[:12]   # 12 hex chars disambiguates
    fn = os.path.join(d, f"{file_stamp(sess)}_{hexid}.md")
    t = sess["tok"]
    cost = sess.get("cost")
    pid = sess.get("priced_id")
    r = sess.get("priced")
    if r:
        rate_note = (f"priced as `{pid}` — ${ (r['prompt'] or 0)*1e6:g}/$"
                     f"{ (r['completion'] or 0)*1e6:g} in/out per Mtok"
                     + (f", ${r['cache_read']*1e6:g} cache-read" if r['cache_read'] is not None else "")
                     + (f", ${r['cache_write']*1e6:g} cache-write" if r['cache_write'] is not None else ""))
        cost_s = f"**${cost:,.2f}**" if cost is not None else "n/a"
    else:
        rate_note = f"no OpenRouter pricing match for model `{', '.join(sess['models']) or '?'}`"
        cost_s = "n/a"
    if sess["agent"] == "claude":
        tokline = (f"- output: **{t['output']:,}**  |  fresh input: {t['input']:,}  |  "
                   f"cache write: {t['cache_create']:,}  |  cache read: {t['cache_read']:,}\n"
                   f"- **cost: {cost_s}** ({rate_note})")
    else:
        tokline = (f"- output: **{t['output']:,}** (reasoning {t['reasoning']:,})  |  "
                   f"input incl. cached: {t['input']:,}  (cached {t['cache_read']:,})  |  "
                   f"total: {t['total']:,}\n"
                   f"- **cost: {cost_s}** ({rate_note})")
    parts = [
        f"# {sess['agent'].title()} session `{sess['sid']}`\n\n",
        f"- **Agent:** {sess['agent']}\n",
        f"- **Model(s):** {', '.join(sess['models']) or '?'}\n",
        f"- **Started:** {fmt_ts(sess['first_ts'])}  |  **Last activity:** {fmt_ts(sess['last_ts'])}\n",
        f"- **Turns:** {sess['n_user']} human / {sess['n_assist']} assistant  |  **Tool calls:** {sess['n_tool']}\n",
        f"- **Source:** `{sess['path']}`\n\n",
        "## Token cost\n\n" + tokline + "\n\n",
        timing_block(sess),
        "## Summary\n\n",
        f"**First request:**\n\n> {trunc(sess['first_prompt'],600).strip().replace(chr(10),chr(10)+'> ')}\n\n",
        f"**Final response:**\n\n> {trunc(sess['last_text'],600).strip().replace(chr(10),chr(10)+'> ')}\n\n",
        "---\n\n## Transcript\n\n",
        sess["transcript"] or "_(no message content)_\n",
    ]
    with open(fn, "w", encoding="utf-8") as f:
        f.write(redact("".join(parts)))   # redact whole file (catches Source path, prompts, tool output)
    return fn

def main():
    global PRICING
    PRICING = load_pricing()
    print("pricing source:", PRICING.get("src"))
    sessions = []
    for p in sorted(glob.glob(os.path.join(CLAUDE_DIR, "*.jsonl"))):
        sessions.append(parse_claude(p))
    for p in glob.glob(CODEX_GLOB, recursive=True):
        # cheap cwd filter
        try:
            head = open(p, encoding="utf-8").read(4000)
        except Exception:
            continue
        if f'"cwd":"{PROJECT}"' in head:
            sessions.append(parse_codex(p))
    sessions.sort(key=lambda s: s["first_ts"] or "")
    # price each session against OpenRouter rates; render timed transcript
    for s in sessions:
        pid, rates = match_pricing(s["models"][0] if s["models"] else None)
        s["priced_id"], s["priced"] = pid, rates
        s["cost"] = session_cost(s)
        s["transcript"], s["timing"] = render_sections(s.get("sections") or [])
    # write per-session files
    rows = []
    for s in sessions:
        fn = write_session(s, s["agent"])   # subdir = "claude" | "codex"
        s["file"] = os.path.relpath(fn, OUT)
        rows.append(s)
    # aggregate
    cl = [s for s in rows if s["agent"]=="claude"]
    cx = [s for s in rows if s["agent"]=="codex"]
    def agg(group):
        return (sum(s["tok"]["output"] for s in group),
                sum((s["cost"] or 0) for s in group),
                all(s["cost"] is not None for s in group if s["n_assist"]))
    cl_out, cl_cost, cl_full = agg(cl)
    cx_out, cx_cost, cx_full = agg(cx)
    priced_models = sorted({f"{s['models'][0]} → `{s['priced_id']}`"
                            for s in rows if s.get("priced_id")})
    R = []
    R.append(f"# Agent sessions for `{PROJECT}`\n\n")
    R.append(f"Collected **{len(cl)} Claude Code** session(s) and **{len(cx)} Codex** session(s) "
             f"whose working directory is this project.\n\n")
    R.append("Per-session transcripts, summaries, and token costs live under `claude/` and `codex/`. "
             "Each file has a header (model, turns, token cost), a summary (first request + final "
             "response), and the full transcript (long tool outputs truncated). Personal name, email, "
             "and OS username are redacted.\n\n")
    R.append("## Aggregate cost\n\n")
    R.append("| Agent | Sessions | Output tokens | Cost (USD) |\n|---|--:|--:|--:|\n")
    R.append(f"| Claude Code | {len(cl)} | {cl_out:,} | **${cl_cost:,.2f}** |\n")
    R.append(f"| Codex | {len(cx)} | {cx_out:,} | **${cx_cost:,.2f}** |\n")
    R.append(f"| **All** | **{len(cl)+len(cx)}** | **{cl_out+cx_out:,}** | **${cl_cost+cx_cost:,.2f}** |\n\n")
    def tsum(group, k):
        return sum((s.get("timing") or {}).get(k, 0) for s in group)
    R.append("## Aggregate time\n\n")
    R.append("| Agent | Wall-clock | Model gen | Tool exec | Active | Waiting for user |\n|---|--:|--:|--:|--:|--:|\n")
    for label, g in (("Claude Code", cl), ("Codex", cx)):
        R.append(f"| {label} | {fmt_dur(tsum(g,'wall'))} | {fmt_dur(tsum(g,'model'))} | "
                 f"{fmt_dur(tsum(g,'tool'))} | {fmt_dur(tsum(g,'model')+tsum(g,'tool'))} | "
                 f"{fmt_dur(tsum(g,'waiting'))} |\n")
    R.append("\n> Each section's time is attributed by what it is: `👤 User`→waiting-for-user, "
             "`🤖 Assistant`→model generation, `🛠️ Tool result`→tool execution; the three tile the "
             "session so they sum to wall-clock. Per-call exec times are matched (`tool_use`↔`tool_result`) "
             "and shown inline on each call line. Codex event timestamps are batch-flushed, so its splits "
             "are approximate.\n\n")
    R.append(f"> **Pricing source:** {PRICING.get('src') or 'unavailable'}. Cost is computed per token "
             "from each model's OpenRouter rates (prompt / completion / cache-read / cache-write), so "
             "cache-read tokens — re-counted every turn — are billed at their reduced rate rather "
             "than inflating the headline. Model rate matches:\n>\n")
    for pm in priced_models:
        R.append(f"> - {pm}\n")
    R.append(">\n> **Caveat on Codex cached tokens (lower bound):** the rollout records the *agent-reported* "
             "`cached_input_tokens`, i.e. how many input tokens Codex *expected* to hit the provider cache. "
             "Actual billing only discounts tokens that genuinely hit the cache (entries expire on a TTL), "
             "so the rest are billed at the full prompt rate. This bites models with a steep cache discount: "
             "e.g. `deepseek-v4-pro` lists cache-read at $0.0036/Mtok, but its real charge here (~$0.36) "
             "implies an effective ~$0.30/Mtok (≈⅓ of the 'cached' tokens actually hit). OpenAI caching "
             "reconciled exactly. Codex costs below are therefore a **lower bound**; Claude (whose cache "
             "reads are reported as billed) is exact.\n")
    R.append("\n")
    for label, group in (("Claude Code", cl), ("Codex", cx)):
        R.append(f"## {label} sessions\n\n")
        R.append("| # | Date | Model | Human/Asst | Tools | Active | Wall | Cost | First request | File |\n")
        R.append("|--|---|---|--:|--:|--:|--:|--:|---|---|\n")
        for i, s in enumerate(group, 1):
            tm = s.get("timing") or {}
            metric = f"${s['cost']:,.2f}" if s["cost"] is not None else "n/a"
            req = (s["first_prompt"] or "").replace("\n"," ").replace("|","\\|")[:50]
            R.append(f"| {i} | {fmt_ts(s['first_ts'])[:16]} | {', '.join(s['models']) or '?'} | "
                     f"{s['n_user']}/{s['n_assist']} | {s['n_tool']} | "
                     f"{fmt_dur(tm.get('model',0)+tm.get('tool',0))} | {fmt_dur(tm.get('wall',0))} | {metric} | "
                     f"{req} | [`{s['file']}`]({s['file']}) |\n")
        R.append("\n")
    with open(os.path.join(OUT, "README.md"), "w", encoding="utf-8") as f:
        f.write(redact("".join(R)))
    print(f"Wrote {len(rows)} sessions to {OUT}")
    print(f"Claude: {len(cl)} sessions · output {cl_out:,} · ${cl_cost:,.2f}")
    print(f"Codex : {len(cx)} sessions · output {cx_out:,} · ${cx_cost:,.2f}")
    print(f"TOTAL : ${cl_cost+cx_cost:,.2f}")

if __name__ == "__main__":
    main()
