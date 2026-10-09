#!/usr/bin/env python3
"""
Agent Autotuner - single-file implementation (PS 04).

Goal: pick tool-calling settings (schema compression level x confidence signal x
escalation threshold) for ONE model and ONE toolset, using synthetic tasks whose
ground truth is known by construction, then check on HELD-OUT tasks whether the
tuned settings beat the pre-registered defaults.

Setup:
    pip install openai numpy matplotlib pyyaml
    set NVIDIA_API_KEY (PowerShell: $env:NVIDIA_API_KEY="nvapi-...")

Run:
    python autotuner.py --list-models --base-url https://integrate.api.nvidia.com/v1
    python autotuner.py --check --base-url https://integrate.api.nvidia.com/v1 --model auto
    python autotuner.py --base-url https://integrate.api.nvidia.com/v1 --model auto
    python autotuner.py --mock                       # code-path test, NO real model

Rules baked in (do not change them after seeing results):
  * DEFAULT_CFG below is the pre-registered default.
  * Settings are selected on the TRAIN split only. TEST is used once, at the end.
  * Verdict says NOT ESTABLISHED unless the CI excludes zero.
"""
import argparse
import json
import math
import os
import random
import re
import sys
import time
import zlib
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Pre-registered settings. Edit BEFORE running, never after seeing results.
# --------------------------------------------------------------------------
DEFAULT_CFG = {"level": "full", "signal": "mean_args", "thr": -1.2}  # hand-picked default
SPEED_WEIGHT = 0.2   # how much a relative latency saving is worth in "yield" points
SIGNALS = ["mean_args", "min_args", "mean_all", "min_all"]

# --------------------------------------------------------------------------
# Toolset (12 tools). Templates contain the argument values verbatim, so the
# expected call is known by construction. "amb" templates omit required info:
# the correct behaviour there is to ask, not to call.
# --------------------------------------------------------------------------
POOLS = {
    "path": ["notes/todo.txt", "reports/q3.md", "src/main.py", "data/users.csv",
             "docs/readme.md", "logs/app.log", "config/settings.yaml", "drafts/essay.txt"],
    "newpath": ["archive/todo_old.txt", "reports/q3_final.md", "src/app.py", "data/users_v2.csv",
                "docs/guide.md", "logs/app_old.log", "config/prod.yaml", "drafts/essay_v2.txt"],
    "dir": ["docs/", "src/", "reports/", "archive/", "data/"],
    "query": ["invoice", "budget", "deadline", "roadmap", "contract"],
    "email": ["priya@acme.com", "bob@x.com", "dev@startup.io", "hr@corp.in", "sam@mail.com", "lee@site.org"],
    "subject": ["Meeting moved", "Invoice attached", "Project update", "Quick question", "Lunch tomorrow"],
    "body": ["See you at the new time", "Please review the file", "Let me know your thoughts", "Thanks for the help"],
    "title": ["Design review", "Team sync", "Dentist", "Demo day", "Budget meeting"],
    "date": ["2026-11-03", "2026-12-15", "2026-10-30", "2027-01-08", "2026-11-21"],
    "time": ["09:00", "14:30", "16:45", "11:15", "18:00"],
    "evt": ["evt_101", "evt_204", "evt_377", "evt_512", "evt_698"],
    "city": ["Bengaluru", "Mumbai", "Delhi", "Chennai", "Pune", "Hyderabad"],
    "rem": ["call the bank", "submit the report", "water the plants", "book the tickets"],
    "phrase": ["Good morning everyone", "Where is the station", "Thank you very much", "See you tomorrow"],
    "lang": ["French", "Spanish", "German", "Hindi"],
}


def P(typ, desc, pool):
    return (typ, desc, pool)


SPEC = {
    "read_file": dict(
        desc="Read the full text content of a file at the given path", destructive=False,
        params={"path": P("string", "File path", "path")},
        templates=["Read the file {path}", "Show me what is inside {path}", "Open {path} and read it to me"],
        amb=["Read that file for me", "Show me the file contents", "Open the document from earlier"]),
    "write_file": dict(
        desc="Write text content to a file, replacing any existing content", destructive=True,
        params={"path": P("string", "File path", "path"), "content": P("string", "Text to write", "body")},
        templates=['Write "{content}" into {path}', 'Save the text "{content}" to {path}'],
        amb=["Write something into the notes file", "Save my text to a file", "Overwrite {path} with new text"]),
    "delete_file": dict(
        desc="Permanently delete a file", destructive=True,
        params={"path": P("string", "File path", "path")},
        templates=["Delete the file {path}", "Remove {path} permanently", "Please erase {path}"],
        amb=["Delete the old file", "Remove that file", "Get rid of the log"]),
    "list_dir": dict(
        desc="List the files inside a directory", destructive=False,
        params={"dir": P("string", "Directory path", "dir")},
        templates=["List everything in {dir}", "What files are in {dir}", "Show the contents of folder {dir}"],
        amb=["List the files", "What is in that folder", "Show me the directory"]),
    "search_files": dict(
        desc="Search files for a keyword inside a directory", destructive=False,
        params={"query": P("string", "Keyword", "query"), "dir": P("string", "Directory path", "dir")},
        templates=['Search {dir} for "{query}"', 'Find files in {dir} that mention "{query}"'],
        amb=["Search the files for something", 'Find files mentioning "{query}"', "Look through the folder"]),
    "rename_file": dict(
        desc="Rename or move a file from an old path to a new path", destructive=True,
        params={"old": P("string", "Current path", "path"), "new": P("string", "New path", "newpath")},
        templates=["Rename {old} to {new}", "Move {old} to {new}"],
        amb=["Rename the file", "Move the report somewhere else", "Rename {old}"]),
    "send_email": dict(
        desc="Send an email to a recipient", destructive=True,
        params={"to": P("string", "Recipient email address", "email"), "subject": P("string", "Email subject", "subject"),
                "body": P("string", "Email body text", "body")},
        templates=['Email {to} with subject "{subject}" and message "{body}"',
                   'Send {to} an email titled "{subject}" saying "{body}"'],
        amb=["Send an email to {to}", "Email someone about the update", "Send a message to the team"]),
    "create_event": dict(
        desc="Create a calendar event", destructive=False,
        params={"title": P("string", "Event title", "title"), "date": P("string", "Date, YYYY-MM-DD", "date"),
                "time": P("string", "Start time, HH:MM", "time")},
        templates=['Schedule "{title}" on {date} at {time}', 'Add a calendar event called "{title}" for {date} at {time}'],
        amb=["Schedule a meeting", 'Add "{title}" to my calendar', "Put something on the calendar for next week"]),
    "delete_event": dict(
        desc="Delete a calendar event by its id", destructive=True,
        params={"event_id": P("string", "Event id", "evt")},
        templates=["Delete calendar event {event_id}", "Cancel the event with id {event_id}"],
        amb=["Delete my next meeting", "Cancel that event", "Remove the appointment"]),
    "get_weather": dict(
        desc="Get the current weather for a city", destructive=False,
        params={"city": P("string", "City name", "city")},
        templates=["What is the weather in {city}", "How is the weather in {city} right now", "Check the weather for {city}"],
        amb=["What is the weather like", "Do I need an umbrella", "How hot is it outside"]),
    "set_reminder": dict(
        desc="Set a reminder with text at a given time", destructive=False,
        params={"text": P("string", "Reminder text", "rem"), "time": P("string", "Time, HH:MM", "time")},
        templates=["Remind me to {text} at {time}", "Set a reminder at {time} to {text}"],
        amb=["Remind me about something", "Set a reminder for later", "Remind me to {text}"]),
    "translate_text": dict(
        desc="Translate text into a target language", destructive=False,
        params={"text": P("string", "Text to translate", "phrase"), "lang": P("string", "Target language", "lang")},
        templates=['Translate "{text}" into {lang}', 'How do you say "{text}" in {lang}'],
        amb=["Translate this for me", 'Translate "{text}"', "Put that sentence in another language"]),
}


def tool_json(name):
    s = SPEC[name]
    return {"name": name, "description": s["desc"],
            "parameters": {"type": "object",
                           "properties": {k: {"type": v[0], "description": v[1]} for k, v in s["params"].items()},
                           "required": list(s["params"])}}


def render_tools(level):
    if level == "full":
        return json.dumps([tool_json(n) for n in SPEC], indent=2)
    lines = []
    for n, s in SPEC.items():
        sig = ", ".join(k + ":" + v[0] for k, v in s["params"].items())
        lines.append(f"{n}({sig})" + (f" - {s['desc']}" if level == "mid" else ""))
    return "\n".join(lines)


def system_prompt(level):
    return ("You are a tool-calling agent. Available tools:\n" + render_tools(level) +
            '\n\nRespond ONLY with JSON: {"tool": "<name>", "args": {...}}. '
            'If required information is missing, respond {"tool": null, "args": {}, "ask": "<question>"}.')


# --------------------------------------------------------------------------
# Task synthesizer + train/test split (ground truth by construction)
# --------------------------------------------------------------------------
def make_tasks(per_tool, seed):
    rng = random.Random(seed)
    n_amb = max(1, per_tool // 5)
    tasks, seen = [], set()
    for name, s in SPEC.items():
        for kind, target in (("normal", per_tool - n_amb), ("amb", n_amb)):
            got = tries = 0
            while got < target and tries < 400:
                tries += 1
                args = {k: rng.choice(POOLS[p[2]]) for k, p in s["params"].items()}
                tpl = rng.choice(s["templates"] if kind == "normal" else s["amb"])
                text = tpl.format(**args)
                if text in seen:
                    continue
                seen.add(text)
                tasks.append({"id": len(tasks), "text": text, "amb": kind == "amb",
                              "tool": name if kind == "normal" else None,
                              "args": args if kind == "normal" else {},
                              "destructive": s["destructive"], "intended": name})
                got += 1
    by_tool = {}
    for t in tasks:
        by_tool.setdefault(t["intended"], []).append(t)
    train, test = [], []
    for name, ts in by_tool.items():
        rng.shuffle(ts)
        for i, t in enumerate(ts):
            (train if i % 2 == 0 else test).append(t)
    return tasks, train, test


# --------------------------------------------------------------------------
# Model calls (real + mock)
# --------------------------------------------------------------------------
_VARIANT = [0]
MAX_TOKENS = 200


def real_call(client, model, system, user):
    base = dict(model=model, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0, max_tokens=MAX_TOKENS)
    variants = [dict(logprobs=True, top_logprobs=1, response_format={"type": "json_object"}),
                dict(logprobs=True, top_logprobs=1),
                dict()]
    last = None
    for vi in range(_VARIANT[0], len(variants)):
        for attempt in range(5):
            try:
                t0 = time.perf_counter()
                r = client.chat.completions.create(**base, **variants[vi])
                dt = time.perf_counter() - t0
                _VARIANT[0] = vi
                c = r.choices[0]
                lp = c.logprobs.content if (getattr(c, "logprobs", None) and c.logprobs.content) else []
                return {"text": c.message.content or "", "tokens": [t.token for t in lp],
                        "logprobs": [max(t.logprob, -50.0) for t in lp],
                        "prompt_tokens": r.usage.prompt_tokens, "completion_tokens": r.usage.completion_tokens,
                        "latency": dt}
            except Exception as e:
                last = e
                msg = str(e).lower()
                if any(code in msg for code in ("error code: 401", "error code: 403", "error code: 404", "error code: 410")):
                    raise RuntimeError(str(e))  # model gone or key rejected: retrying will not help
                if "429" in msg or "rate" in msg or "timeout" in msg:
                    time.sleep(3 * (attempt + 1))  # rate limit / slow hosted endpoint: back off, same variant
                    continue
                break  # request rejected: try a simpler variant
    raise RuntimeError(f"model call failed: {last}")


def mock_call(system, task, level):
    """Simulated model. Used ONLY to test the code path. Results are meaningless."""
    rng = random.Random(zlib.crc32(f"{task['id']}-{level}".encode()))
    p_ok = {"full": 0.90, "mid": 0.86, "min": 0.82}.get(level, 0.85)
    wrong = False
    if task["amb"]:
        if rng.random() < 0.7:
            obj = {"tool": None, "args": {}, "ask": "Which one?"}
        else:
            s = SPEC[task["intended"]]
            obj = {"tool": task["intended"], "args": {k: rng.choice(POOLS[p[2]]) for k, p in s["params"].items()}}
            wrong = True
    else:
        obj = {"tool": task["tool"], "args": dict(task["args"])}
        if rng.random() > p_ok:
            wrong = True
            if rng.random() < 0.5:
                other = rng.choice([n for n in SPEC if n != task["tool"]])
                s = SPEC[other]
                obj = {"tool": other, "args": {k: rng.choice(POOLS[p[2]]) for k, p in s["params"].items()}}
            else:
                k = rng.choice(list(obj["args"]))
                obj["args"][k] = rng.choice(POOLS[SPEC[task["tool"]]["params"][k][2]]) + "x"
    text = json.dumps(obj)
    tokens = [text[i:i + 3] for i in range(0, len(text), 3)]
    lps = [-abs(rng.gauss(1.4, 0.9)) if wrong else -abs(rng.gauss(0.15, 0.15)) for _ in tokens]
    ptok = (len(system) + len(task["text"])) // 4
    return {"text": text, "tokens": tokens, "logprobs": lps, "prompt_tokens": ptok,
            "completion_tokens": len(tokens), "latency": 0.15 + ptok * 0.0006 + rng.uniform(0, 0.05)}


# --------------------------------------------------------------------------
# Model discovery (models get retired: always ask the API what is live)
# --------------------------------------------------------------------------
EXCLUDE = re.compile(r"embed|rerank|vision|-vl|vlm|guard|safety|parse|reward|retriev|clip|tts|asr|image|"
                     r"diffusion|flux|bge|e5-|reasoning|-r1|think|code|translate|riva|nvclip", re.I)
PREFERRED = [r"llama-3\.1-8b-instruct", r"llama-3\.2-3b", r"nemotron-mini-4b", r"mistral-7b-instruct",
             r"qwen.*-7b", r"gemma.*(2b|7b|9b)", r"phi-3.*mini", r"nemotron-nano-8b", r"8b", r"7b", r"nano", r"mini"]


def list_models(client):
    try:
        return sorted(m.id for m in client.models.list().data)
    except Exception as e:
        raise RuntimeError(f"could not list models: {e}")


SMALL_FIRST = ["google/gemma-3-4b-it", "ibm/granite-3.0-3b-a800m-instruct", "google/gemma-3-12b-it",
               "ibm/granite-3.0-8b-instruct", "nvidia/mistral-nemo-minitron-8b-8k-instruct",
               "nv-mistralai/mistral-nemo-12b-instruct", "mistralai/mistral-7b-instruct-v0.3",
               "meta/llama-3.1-8b-instruct", "google/gemma-2b", "zyphra/zamba2-7b-instruct",
               "nvidia/nemotron-nano-3-30b-a3b", "nvidia/nemotron-3.5-lightning-30b-a3b",
               "google/gemma-4-31b-it", "openai/gpt-oss-20b", "nvidia/llama-3.1-nemotron-70b-instruct"]


def probe_models(client, ids, limit=15):
    """Listed does not mean callable: try each candidate with one tiny real request."""
    served = set(ids)
    order = [m for m in SMALL_FIRST if m in served]
    order += [m for m in ids if m not in order and not EXCLUDE.search(m)]
    order = order[:limit]
    task = {"id": -1, "text": "Email bob@x.com with subject \"Hi\" and message \"Hello there\"", "amb": False,
            "tool": "send_email", "args": {"to": "bob@x.com", "subject": "Hi", "body": "Hello there"},
            "destructive": True, "intended": "send_email"}
    sysmsg = system_prompt("min")
    results = []
    print(f"Probing {len(order)} models with one tiny request each...")
    for m in order:
        _VARIANT[0] = 0
        try:
            out = real_call(client, m, sysmsg, task["text"])
            row = build_row(task, out)
            res = (m, True, len(out["logprobs"]), not row["failed"], out["latency"], "")
            print(f"  OK    logprobs={res[2]:<3} json={'OK ' if res[3] else 'BAD'} {res[4]:.1f}s  {m}")
        except RuntimeError as e:
            res = (m, False, 0, False, 0.0, str(e)[:80])
            print(f"  FAIL  {m}   {res[5]}")
        results.append(res)
    return results


def pick_model(ids):
    ok = [i for i in ids if not EXCLUDE.search(i)]
    for pat in PREFERRED:
        hits = sorted((i for i in ok if re.search(pat, i, re.I)), key=len)
        if hits:
            return hits[0]
    raise RuntimeError("no suitable small instruct model found; run --list-models and pass --model <id>")


# --------------------------------------------------------------------------
# Parsing, scoring, confidence signals
# --------------------------------------------------------------------------
def parse_call(text):
    try:
        obj = json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text or "", re.S)
        try:
            obj = json.loads(m.group(0)) if m else None
        except Exception:
            obj = None
    return obj if isinstance(obj, dict) else None


def norm(v):
    return str(v).strip().strip("\"'").lower()


def conf_signals(tokens, logprobs):
    if not logprobs:
        return {}
    joined = "".join(tokens)
    m = joined.find('"args"')
    start = m + len('"args"') if m >= 0 else 0
    pos, sel = 0, []
    for tok, lp in zip(tokens, logprobs):
        if pos >= start:
            sel.append(lp)
        pos += len(tok)
    if not sel:
        sel = list(logprobs)
    return {"mean_args": float(np.mean(sel)), "min_args": float(np.min(sel)),
            "mean_all": float(np.mean(logprobs)), "min_all": float(np.min(logprobs))}


def build_row(task, out):
    obj = parse_call(out["text"])
    failed = obj is None
    tool = None if failed else obj.get("tool")
    if isinstance(tool, str) and tool.strip().lower() in ("", "null", "none"):
        tool = None
    args = obj.get("args") if (obj and isinstance(obj.get("args"), dict)) else {}
    asked = failed or tool is None
    if task["amb"]:
        correct = (not failed) and tool is None
    else:
        correct = (not failed) and tool == task["tool"] and all(
            norm(args.get(k, "\u0000")) == norm(v) for k, v in task["args"].items())
    pred_destr = bool(tool in SPEC and SPEC[tool]["destructive"])
    return {"id": task["id"], "text": task["text"], "pred_tool": tool, "pred_args": args,
            "asked": bool(asked), "failed": bool(failed), "correct": bool(correct), "pred_destructive": pred_destr,
            "conf": conf_signals(out["tokens"], out["logprobs"]),
            "prompt_tokens": out["prompt_tokens"], "completion_tokens": out["completion_tokens"],
            "latency": out["latency"], "raw": out["text"]}


# --------------------------------------------------------------------------
# Running + caching
# --------------------------------------------------------------------------
def run_level(level, tasks, args, client, outdir, tag):
    path = outdir / f"rows_{tag}_{level}.jsonl"
    rows = {}
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            rows[r["id"]] = r
    sysmsg = system_prompt(level)
    todo = [t for t in tasks if not (t["id"] in rows and rows[t["id"]]["text"] == t["text"])]
    print(f"[{level}] {len(tasks) - len(todo)} cached, {len(todo)} to run")
    with path.open("a") as f:
        for i, t in enumerate(todo):
            out = mock_call(sysmsg, t, level) if args.mock else real_call(client, args.model, sysmsg, t["text"])
            r = build_row(t, out)
            rows[t["id"]] = r
            f.write(json.dumps(r) + "\n")
            f.flush()
            if (i + 1) % 10 == 0 or i + 1 == len(todo):
                print(f"  [{level}] {i + 1}/{len(todo)}")
    return rows


# --------------------------------------------------------------------------
# Outcomes + metrics
# --------------------------------------------------------------------------
def outcome(row, task, cfg):
    """a=auto-correct, b=silent error, c=escalation that was needed, d=unnecessary escalation, u=unsafe."""
    conf = row["conf"].get(cfg["signal"])
    low = conf is not None and conf < cfg["thr"]
    esc = row["asked"] or low
    a = b = c = d = u = 0
    if task["amb"]:
        if esc:
            c = 1
        else:
            b = 1
    elif esc:
        if row["correct"] or row["asked"]:
            d = 1
        else:
            c = 1
    elif row["correct"]:
        a = 1
    else:
        b = 1
    if b and (row["pred_destructive"] or task["destructive"]):
        u = 1
    return a, b, c, d, u


def vectors(rows, tasks, cfg):
    lvl = rows[cfg["level"]]
    n = len(tasks)
    o = np.zeros((n, 5))
    lat, pt, ct = np.zeros(n), np.zeros(n), np.zeros(n)
    for i, t in enumerate(tasks):
        r = lvl[t["id"]]
        o[i] = outcome(r, t, cfg)
        lat[i], pt[i], ct[i] = r["latency"], r["prompt_tokens"], r["completion_tokens"]
    return {"a": o[:, 0], "b": o[:, 1], "c": o[:, 2], "d": o[:, 3], "u": o[:, 4],
            "lat": lat, "ptok": pt, "tok": pt + ct}


def summarize(v):
    return {"yield": float(v["a"].mean()), "silent": float(v["b"].mean()), "caught": float(v["c"].mean()),
            "unnec": float(v["d"].mean()), "unsafe": float(v["u"].mean()),
            "ptok": float(v["ptok"].mean()), "tok": float(v["tok"].mean()),
            "lat": float(v["lat"].mean()), "lat_p95": float(np.percentile(v["lat"], 95))}


def raw_accuracy(rows, tasks, level):
    return float(np.mean([rows[level][t["id"]]["correct"] for t in tasks]))


def paired_bootstrap(vt, vo, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    N = len(vt["a"])
    idx = rng.integers(0, N, (n, N))
    res = {}
    for name, key in [("yield", "a"), ("silent", "b"), ("unsafe", "u"), ("unnec", "d"), ("lat", "lat"), ("ptok", "ptok")]:
        diff = vt[key] - vo[key]
        boots = diff[idx].mean(axis=1)
        res[name] = (float(diff.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    return res


# --------------------------------------------------------------------------
# Tuner (TRAIN split only)
# --------------------------------------------------------------------------
def thr_candidates(rows, train, level, signal):
    vals = [rows[level][t["id"]]["conf"].get(signal) for t in train]
    vals = [v for v in vals if v is not None]
    cands = {DEFAULT_CFG["thr"], -1e9}
    if vals:
        cands |= set(float(x) for x in np.quantile(vals, np.linspace(0, 1, 41)))
    return sorted(cands)


def tune(rows, train, levels, signals):
    dv = summarize(vectors(rows, train, DEFAULT_CFG))
    best, grid = None, []
    for level in levels:
        for sig in signals:
            for thr in thr_candidates(rows, train, level, sig):
                cfg = {"level": level, "signal": sig, "thr": thr}
                m = summarize(vectors(rows, train, cfg))
                grid.append((cfg, m))
                if m["silent"] > dv["silent"] + 1e-12 or m["unsafe"] > dv["unsafe"] + 1e-12:
                    continue  # constraint: no worse than default on silent errors and unsafe calls
                score = m["yield"] + SPEED_WEIGHT * (1 - m["lat"] / max(dv["lat"], 1e-9))
                key = (score, -m["tok"])
                if best is None or key > best[0]:
                    best = (key, cfg, m)
    return best[1], best[2], dv


def independent_cfg(rows, train, levels):
    """Baseline: pick compression on raw accuracy alone, keep default signal and threshold."""
    best = max(levels, key=lambda lv: (raw_accuracy(rows, train, lv),
                                       -float(np.mean([rows[lv][t["id"]]["prompt_tokens"] for t in train]))))
    return {"level": best, "signal": DEFAULT_CFG["signal"], "thr": DEFAULT_CFG["thr"]}


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def fmt_cfg(c):
    return f"level={c['level']} signal={c['signal']} thr={c['thr']:.3f}"


def print_table(rows_):
    hdr = f"{'config':<14}{'yield':>8}{'silent':>8}{'unsafe':>8}{'unnec':>8}{'ptok':>9}{'lat(s)':>9}{'p95(s)':>9}"
    print(hdr)
    print("-" * len(hdr))
    for name, m in rows_:
        print(f"{name:<14}{m['yield']:>8.3f}{m['silent']:>8.3f}{m['unsafe']:>8.3f}{m['unnec']:>8.3f}"
              f"{m['ptok']:>9.0f}{m['lat']:>9.3f}{m['lat_p95']:>9.3f}")


def print_diff(name, bs):
    print(f"\nTuned minus {name} (held-out, paired bootstrap 95% CI):")
    for k, label in [("yield", "automation yield"), ("silent", "silent errors"), ("unsafe", "unsafe calls"),
                     ("unnec", "unnecessary escalations"), ("lat", "latency (s)"), ("ptok", "prompt tokens")]:
        d, lo, hi = bs[k]
        print(f"  {label:<26}{d:>+9.3f}   [{lo:>+8.3f}, {hi:>+8.3f}]")
    y_lo = bs["yield"][1]
    ok = y_lo > 0 and bs["silent"][0] <= 0 and bs["unsafe"][0] <= 0
    print(f"  PRIMARY CLAIM vs {name}: {'ESTABLISHED' if ok else 'NOT ESTABLISHED'}"
          f"  (needs yield CI > 0 and silent/unsafe point estimates <= 0)")
    return ok


def pareto_front(points):
    pts = sorted(points, key=lambda p: (p[0], -p[1]))
    front, best_y = [], -1
    for x, y in pts:
        if y > best_y:
            front.append((x, y))
            best_y = y
    return front


def make_plot(rows, test, levels, signals, chosen, path, mock):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    colors = {"full": "tab:blue", "mid": "tab:orange", "min": "tab:green"}
    allpts = []
    for level in levels:
        pts = []
        for sig in signals:
            for thr in thr_candidates(rows, test, level, sig):
                m = summarize(vectors(rows, test, {"level": level, "signal": sig, "thr": thr}))
                pts.append((m["unnec"], m["yield"]))
        allpts += pts
        ax[0].scatter(*zip(*pts), s=10, alpha=0.35, color=colors.get(level, "gray"), label=f"level={level}")
    f = pareto_front(allpts)
    ax[0].plot(*zip(*f), color="black", lw=1.2, label="Pareto frontier (held-out)")
    marks = {"default": ("s", "red"), "independent": ("^", "purple"), "tuned": ("*", "gold")}
    for name, (cfg, m) in chosen.items():
        mk, col = marks[name]
        ax[0].scatter([m["unnec"]], [m["yield"]], marker=mk, s=160 if name == "tuned" else 90,
                      color=col, edgecolor="black", zorder=5, label=name)
    ax[0].set_xlabel("unnecessary escalation rate (lower is better)")
    ax[0].set_ylabel("automation yield (higher is better)")
    ax[0].set_title("Yield vs escalation, held-out tasks")
    ax[0].legend(fontsize=7)
    names = list(chosen)
    lat = [chosen[n][1]["lat"] for n in names]
    bars = ax[1].bar(names, lat, color=[marks[n][1] for n in names], edgecolor="black")
    for b, n in zip(bars, names):
        ax[1].text(b.get_x() + b.get_width() / 2, b.get_height(), f"{chosen[n][1]['ptok']:.0f} prompt tok",
                   ha="center", va="bottom", fontsize=8)
    ax[1].set_ylabel("mean latency per task (s)")
    ax[1].set_title("Speed of chosen configs")
    if mock:
        fig.suptitle("MOCK DATA - NOT A RESULT", color="red", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def export_config(tuned, model, test_summary, outdir, mock):
    cfg = {"model": model, "tool_schema_level": tuned["level"], "confidence_signal": tuned["signal"],
           "escalate_if_below": round(tuned["thr"], 4), "selected_on": "train split",
           "heldout_summary": {k: round(v, 4) for k, v in test_summary.items()},
           "mock": bool(mock)}
    try:
        import yaml
        (outdir / "tuned_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    except ImportError:
        (outdir / "tuned_config.json").write_text(json.dumps(cfg, indent=2))
    (outdir / "tuned_tools_prompt.txt").write_text(system_prompt(tuned["level"]))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def smoke_check(args, client):
    t = {"id": -1, "text": "Email bob@x.com with subject \"Meeting moved\" and message \"See you at the new time\"",
         "amb": False, "tool": "send_email",
         "args": {"to": "bob@x.com", "subject": "Meeting moved", "body": "See you at the new time"},
         "destructive": True, "intended": "send_email"}
    lp_ok, json_ok = True, True
    for level in args.levels:
        sysmsg = system_prompt(level)
        try:
            out = mock_call(sysmsg, t, level) if args.mock else real_call(client, args.model, sysmsg, t["text"])
        except RuntimeError as e:
            print(f"[{level}] FAILED: {e}")
            print("\nIf this says 404/410 (model gone), run: python autotuner.py --list-models "
                  "--base-url https://integrate.api.nvidia.com/v1  and pass a live ID with --model.")
            return
        row = build_row(t, out)
        lp_ok = lp_ok and len(out["logprobs"]) > 0
        json_ok = json_ok and not row["failed"]
        print(f"[{level}] prompt_tokens={out['prompt_tokens']} latency={out['latency']:.2f}s "
              f"logprob_tokens={len(out['logprobs'])} correct={row['correct']}")
        print("   output:", out["text"].strip()[:200])
        print("   confidence signals:", {k: round(v, 3) for k, v in row["conf"].items()})
    print(f"\nMODEL: {args.model}")
    print("LOGPROBS:", "OK" if lp_ok else "MISSING (confidence gating disabled; only compression can be tuned)")
    print("JSON OUTPUT:", "OK" if json_ok else "BROKEN (raise --max-tokens or pick a non-reasoning model)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="auto", help="model id, or auto to pick a small live instruct model")
    ap.add_argument("--probe", action="store_true", help="test which models actually respond, then exit")
    ap.add_argument("--probe-limit", type=int, default=15, help="max models to probe")
    ap.add_argument("--list-models", action="store_true", help="print model ids the API serves, then exit")
    ap.add_argument("--filter", default="", help="substring filter for --list-models")
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--levels", default="full,min", help="comma list from: full,mid,min")
    ap.add_argument("--per-tool", type=int, default=12, help="tasks per tool (20% are ambiguous)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-tokens", type=int, default=200, help="raise if outputs look truncated")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--check", action="store_true", help="smoke test only")
    ap.add_argument("--mock", action="store_true", help="simulated model, tests the code path only")
    args = ap.parse_args()
    globals()["MAX_TOKENS"] = args.max_tokens
    args.levels = [x.strip() for x in args.levels.split(",") if x.strip()]
    if DEFAULT_CFG["level"] not in args.levels:
        args.levels.insert(0, DEFAULT_CFG["level"])

    client = None
    if args.mock:
        print("=" * 70 + "\nMOCK MODE: simulated model. Numbers below are NOT results.\n" + "=" * 70)
        args.out = args.out if args.out != "runs" else "runs_mock"
    else:
        from openai import OpenAI
        key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("OPENAI_API_KEY") or "local"
        client = OpenAI(base_url=args.base_url, api_key=key, timeout=120)
        if args.list_models or args.probe or args.model == "auto":
            try:
                ids = list_models(client)
            except RuntimeError as e:
                sys.exit(str(e))
            if args.list_models:
                for i in ids:
                    if args.filter.lower() in i.lower():
                        print(i)
                print(f"\n{len(ids)} models served")
                return
            res = probe_models(client, ids, args.probe_limit)
            if args.probe:
                return
            good = [r for r in res if r[1] and r[3] and r[2] > 0] or [r for r in res if r[1] and r[3]]
            if not good:
                sys.exit("No probed model answered with valid JSON. Try --probe-limit 40 or pass --model <id>.")
            args.model = good[0][0]
            print("auto-selected model:", args.model, "(first that answered with valid JSON"
                  + (" and logprobs)" if good[0][2] > 0 else "; NO logprobs available)"))

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.check:
        smoke_check(args, client)
        return

    tasks, train, test = make_tasks(args.per_tool, args.seed)
    print(f"{len(tasks)} tasks: {len(train)} train / {len(test)} held-out; "
          f"{sum(t['amb'] for t in tasks)} ambiguous")
    tag = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{'mock' if args.mock else args.model}_s{args.seed}_n{args.per_tool}")

    rows = {lv: run_level(lv, tasks, args, client, outdir, tag) for lv in args.levels}

    have_conf = any(r["conf"] for lv in rows.values() for r in lv.values())
    signals = SIGNALS if have_conf else ["none"]
    if not have_conf:
        print("\nWARNING: no logprobs returned. Confidence gating disabled; only compression is tuned.")

    tuned_cfg, tuned_train, default_train = tune(rows, train, args.levels, signals)
    indep_cfg = independent_cfg(rows, train, args.levels)
    print("\nSelected on TRAIN only:")
    print("  default    :", fmt_cfg(DEFAULT_CFG))
    print("  independent:", fmt_cfg(indep_cfg))
    print("  tuned      :", fmt_cfg(tuned_cfg))

    cfgs = {"default": DEFAULT_CFG, "independent": indep_cfg, "tuned": tuned_cfg}
    vt = {n: vectors(rows, test, c) for n, c in cfgs.items()}
    sm = {n: summarize(v) for n, v in vt.items()}

    print(f"\nHELD-OUT results ({len(test)} tasks, evaluated once):")
    print_table([(n, sm[n]) for n in cfgs])
    ok_def = print_diff("default", paired_bootstrap(vt["tuned"], vt["default"]))
    ok_ind = print_diff("independent", paired_bootstrap(vt["tuned"], vt["independent"]))
    print("\nRaw (ungated) accuracy per compression level, held-out: " +
          ", ".join(f"{lv}={raw_accuracy(rows, test, lv):.3f}" for lv in args.levels))
    print("\nHonest reading: a test set this small gives wide intervals. If a claim says NOT ESTABLISHED, "
          "do not pitch it as an optimizer win; present the trade-off frontier instead.")
    if args.mock:
        print("\n(MOCK RUN: ignore every number above.)")

    make_plot(rows, test, args.levels, signals, {n: (cfgs[n], sm[n]) for n in cfgs}, outdir / "pareto.png", args.mock)
    export_config(tuned_cfg, "mock" if args.mock else args.model, sm["tuned"], outdir, args.mock)
    (outdir / "results.json").write_text(json.dumps({
        "mock": args.mock, "model": args.model, "n_train": len(train), "n_test": len(test),
        "configs": cfgs, "train_default": default_train, "train_tuned": tuned_train, "heldout": sm,
        "claim_vs_default": ok_def, "claim_vs_independent": ok_ind}, indent=2))
    print(f"\nSaved to {outdir}/: pareto.png, tuned_config.yaml, tuned_tools_prompt.txt, results.json, rows_*.jsonl")


if __name__ == "__main__":
    main()