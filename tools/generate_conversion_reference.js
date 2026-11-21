#!/usr/bin/env node
/**
 * Maintainer-only tool: regenerate the HMVC conversion-reference fixture.
 *
 * Runs the ORIGINAL DSP code from ``voice-converter (4).html`` (extracted
 * verbatim, never modified) in Node on the fixed synthetic source/target in
 * ``tests/data/conversion-reference/signals.json`` and writes the outputs to
 * ``tests/data/conversion-reference/reference.json``.  The HMVC port is then
 * compared against this fixture (tests/conversion/test_reference.py) without
 * needing a browser.
 *
 * Usage:
 *   node tools/generate_conversion_reference.js
 *
 * The harness:
 *   1. reads the HTML, extracts the <script> body up to the UI section,
 *   2. imports it unmodified into a function scope (the DSP code is never
 *      edited — only one instrumentation assignment inside `vocode`, which
 *      captures internal arrays immediately before its original final
 *      `return out;`, is inserted),
 *   3. monkeypatches Math.random with a deterministic 64-bit LCG (the same
 *      constants as hmvc.conversion.rng) so the vocoder's noise is
 *      reproducible — the browser uses an unseeded Math.random, which can
 *      never be reproduced exactly by another implementation,
 *   4. re-drives the same control flow as the original convert() with no UI.
 *
 * The driver (UI-free mirror of convert()) is written in this file and mirrors
 * ``src/hmvc/conversion/pipeline.py`` — keep them in sync.  This tool uses
 * only Node stdlib; it never imports hmvc.
 */

const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "..");
const HTML = path.join(ROOT, "voice-converter (4).html");
const REF_DIR = path.join(ROOT, "tests", "data", "conversion-reference");
const SIGNALS = path.join(REF_DIR, "signals.json");
const OUTPUT = path.join(REF_DIR, "reference.json");

// ---------------------------------------------------------------------------
// 1. extracted original JS ---------------------------------------------------
// ---------------------------------------------------------------------------

const html = fs.readFileSync(HTML, "utf8");
const scriptStart = html.indexOf("<script>");
const scriptEnd = html.indexOf("</script>");
if (scriptStart < 0 || scriptEnd < 0) throw new Error("HTML script block not found");
let js = html.slice(scriptStart + "<script>".length, scriptEnd);
const uiMarker = js.indexOf("/* ---------- UI ---------- */");
if (uiMarker < 0) throw new Error("UI marker not found in extracted script");
js = js.slice(0, uiMarker);

for (const fn of ["trackPitch", "fillGaps", "median", "makeMarks", "psola",
                  "analyze", "feat", "cmn", "buildDB", "knn", "grainSynth",
                  "inv", "vocode", "fft"]) {
    if (!js.includes(fn + "(")) throw new Error("extracted JS is missing " + fn);
}

// Instrumentation only: capture vocode internals before its original final
// `return out;`.  No DSP arithmetic is changed, only variables read.
const marker = "  return out;\n}\n\n/* ---------- main conversion";
if (!js.includes(marker)) throw new Error("vocode return marker not found");
js = js.replace(
    marker,
    "  globalThis.__debug = { act, X, Y, M, mu, vr, pi, G, MX, MY, AA, P, COS, Am, F0, HM, PH, out: out.slice(0, len) };\n  return out;\n}\n\n/* ---------- main conversion"
);
// instrument right after the per-frame loop (before PH) with the noise-only out
const marker2 = "  const PH=new Float32Array(NH+1);";
if (!js.includes(marker2)) throw new Error("PH marker not found");
js = js.replace(
    marker2,
    "  globalThis.__noise = out.slice(0, len).slice();\n  const PH=new Float32Array(NH+1);"
);

// ---------------------------------------------------------------------------
// 2. deterministic Math.random (same constants as hmvc.conversion.rng) ------
// ---------------------------------------------------------------------------
const LCG = (() => {
    let state = 0n;
    const MASK = 0xffffffffffffffffn;
    const MULT = 0x5851f42d4c957f2dn; // 6364136223846793005
    const INC = 0x14057b7ef767814fn; // 1442695040888963407
    return (seed) => {
        state = BigInt(seed >>> 0) & MASK;
        if (state === 0n) state = 1n;
        return function () {
            state = (state * MULT + INC) & MASK;
            return Number(state >> 11n) * (1.0 / 9007199254740992.0);
        };
    };
})();

Math.random = LCG(1234);

// ---------------------------------------------------------------------------
// 3. load the original functions unmodified ----------------------------------
// ---------------------------------------------------------------------------
const original = new Function(
    js +
    "; return { SR, N, H, NB, HOP, D, NM, LIF, SIL, win: win, fft, trackPitch, fillGaps, median, " +
    "makeMarks, psola, analyze, feat, cmn, buildDB, knn, grainSynth, inv, vocode, Re, Im };"
)();

const { N, H, NB, HOP, D, SIL, win, fft, trackPitch, fillGaps, median, makeMarks,
        psola, analyze, feat, cmn, buildDB, knn, grainSynth, vocode, Re, Im } = original;

// ---------------------------------------------------------------------------
// 4. driver — UI-free mirror of the original convert() control flow ---------
// ---------------------------------------------------------------------------
async function runCase(name, params) {
    const x = params.x;
    const P = { semi: params.semi, form: params.form, ret: params.ret / 100, K: params.K };
    const fmin = params.fmin, fmax = params.fmax;
    const md = params.mode;
    const auto = params.auto;

    const progress = () => {};

    const [f0r, vor] = await trackPitch(x, fmin, fmax, progress);
    const [f0, vo] = fillGaps(f0r, vor);

    let db = null;
    if (params.target && (P.ret > 0 || md !== "env")) {
        const t = params.target;
        const dbNew = await buildDB(t, fmin, fmax, progress);
        db = (dbNew && dbNew.n >= 30) ? dbNew : null;
    }
    const grain = !!db && md === "grain";
    const voc = !!db && md === "voc";
    const direct = grain || voc;

    let ratio = 2 ** (P.semi / 12);
    if (db && auto && db.med) {
        const sm = median(f0, vo);
        if (sm) ratio *= db.med / sm;
    }
    ratio = Math.max(0.4, Math.min(3, ratio));
    const y = (Math.abs(ratio - 1) > 0.005 && !direct) ? psola(x, makeMarks(x, f0, vo), ratio) : x.slice();

    const nfr = Math.floor((y.length + N / 2) / H) + 1;
    const le = new Float32Array(NB);
    const sv = new Uint8Array(nfr);
    const sa = new Uint8Array(nfr);
    const rs = new Float32Array(nfr);
    let nb = null, sf = null;

    if (db) {
        sf = new Float32Array(nfr * D);
        for (let f = 0; f < nfr; f++) {
            const p = f * H - N / 2;
            rs[f] = analyze(y, p, le);
            sa[f] = rs[f] >= SIL ? 1 : 0;
            feat(le, sf, f * D);
            sv[f] = vo[Math.min(vo.length - 1, Math.max(0, Math.round((p + N / 2) / HOP)))];
        }
        cmn(sf, sa);
        {
            const sm = sf.slice();
            for (let f = 1; f < nfr - 1; f++) for (let q = 0; q < D; q++)
                sf[f * D + q] = 0.25 * sm[(f - 1) * D + q] + 0.5 * sm[f * D + q] + 0.25 * sm[(f + 1) * D + q];
        }
        nb = await knn(sf, sv, sa, db, P.K, progress);
    } else {
        for (let f = 0; f < nfr; f++)
            sv[f] = vo[Math.min(vo.length - 1, Math.max(0, Math.round((f * H) / HOP)))];
    }

    let o, tl, ret, prev, have = false;
    if (grain) {
        o = grainSynth(y.length, f0, vo, nb, P.K, db, ratio, rs, nfr, sf);
    } else if (voc) {
        o = await vocode(y.length, f0, vo, nb, P.K, db, ratio, rs, sf, sa, nfr, P.form, (p, msg) => {});
    } else {
        o = new Float32Array(y.length + N);
    }
    tl = new Float32Array(NB); ret = new Float32Array(NB); prev = new Float32Array(NB);

    for (let f = 0; f < (direct ? 0 : nfr); f++) {
        const p = f * H - N / 2, rms = analyze(y, p, le);
        if (rms >= SIL) {
            let mixed = le;
            if (nb && nb[f * P.K] >= 0 && P.ret > 0) {
                ret.fill(0); let c = 0;
                for (let q = 0; q < P.K; q++) {
                    const j = nb[f * P.K + q]; if (j < 0) continue; c++;
                    const e = db.le[j]; for (let k = 0; k < NB; k++) ret[k] += e[k];
                }
                let ms = 0, mr = 0;
                for (let k = 0; k < NB; k++) { ret[k] /= c; ms += le[k]; mr += ret[k]; }
                const off = (ms - mr) / NB;
                for (let k = 0; k < NB; k++) {
                    let r = ret[k] + off; if (have) r = 0.6 * r + 0.4 * prev[k]; prev[k] = r;
                    tl[k] = le[k] * (1 - P.ret) + r * P.ret;
                }
                have = true; mixed = tl;
            }
            for (let k = 0; k < NB; k++) {
                let t = mixed[k];
                if (P.form !== 1) {
                    const u = k / P.form, i = Math.min(NB - 2, Math.floor(u)), a = Math.min(1, u - i);
                    t = mixed[i] * (1 - a) + mixed[i + 1] * a;
                }
                let d = Math.max(-3, Math.min(3, t - le[k])); if (!sv[f]) d *= 0.6;
                const g = Math.exp(d); Re[k] *= g; Im[k] *= g;
                if (k > 0 && k < N / 2) { Re[N - k] *= g; Im[N - k] *= g; }
            }
        }
        fft(Re, Im, 1);
        for (let i = 0; i < N; i++) { const q = p + i; if (q >= 0 && q < o.length) o[q] += Re[i] * win[i] / 1.5; }
    }

    let pk = 0;
    const res = o.slice(0, y.length);
    for (let i = 0; i < res.length; i++) { if (!(res[i] === res[i])) res[i] = 0; pk = Math.max(pk, Math.abs(res[i])); }
    const gn = pk > 0 ? 0.92 / pk : 1;
    for (let i = 0; i < res.length; i++) res[i] *= gn;

    // vocode debug internals (instrumentation only; null for other modes)
    let debug = null;
    if (globalThis.__debug) {
        const d = globalThis.__debug;
        debug = {
            act: Array.from(d.act), M: d.M,
            X: Array.from(d.X), Y: Array.from(d.Y),
            mu: Array.from(d.mu), vr: Array.from(d.vr), pi: Array.from(d.pi),
            G: Array.from(d.G),
            MX: d.MX.map((a) => Array.from(a)),
            MY: d.MY.map((a) => Array.from(a)),
            AA: d.AA.map((a) => Array.from(a)),
            P: Array.from(d.P),
            F0: Array.from(d.F0), HM: Array.from(d.HM), PH: Array.from(d.PH),
            Am: Array.from(d.Am), out: Array.from(d.out),
            noise: globalThis.__noise ? Array.from(globalThis.__noise) : null,
        };
        delete globalThis.__debug;
    }

    return {
        name,
        ratio,
        y_length: y.length,
        f0: Array.from(f0),
        vo: Array.from(vo),
        nfr,
        rs: Array.from(rs),
        sa: Array.from(sa),
        sv: Array.from(sv),
        sf: sf ? Array.from(sf) : null,
        nb: nb ? Array.from(nb) : null,
        db: db ? {
            n: db.n,
            med: db.med,
            idx_0: db.idx[0].length,
            idx_1: db.idx[1].length,
            cvar_mean: db.cvar ? Array.from(db.cvar).reduce((a, b) => a + b, 0) / db.cvar.length : null,
            cvar: Array.from(db.cvar),
        } : null,
        debug,
        samples: Array.from(res),
    };
}

// ---------------------------------------------------------------------------
// 5. generate ---------------------------------------------------------------
// ---------------------------------------------------------------------------
async function main() {
    if (!fs.existsSync(SIGNALS)) throw new Error("missing signals.json — generate it first");
    const signals = JSON.parse(fs.readFileSync(SIGNALS, "utf8"));
    if (signals.sample_rate !== 22050) throw new Error("signals.json sample_rate must be 22050");

    const x = new Float32Array(signals.signals.source);
    const tgt = new Float32Array(signals.signals.target);

    const cases = [
        { name: "env-no-target", params: { mode: "env", semi: 3.0, form: 1.0, ret: 0, K: 6, fmin: 70, fmax: 450, auto: true, x, target: null } },
        { name: "env-with-target", params: { mode: "env", semi: 3.0, form: 1.15, ret: 80, K: 6, fmin: 70, fmax: 450, auto: true, x, target: tgt } },
        { name: "grain", params: { mode: "grain", semi: 0.0, form: 1.0, ret: 80, K: 6, fmin: 70, fmax: 450, auto: true, x, target: tgt } },
        { name: "voc", params: { mode: "voc", semi: 0.0, form: 1.0, ret: 80, K: 6, fmin: 70, fmax: 450, auto: true, x, target: tgt } },
    ];

    const reference = {
        generated_by: "tools/generate_conversion_reference.js",
        source_file: "voice-converter (4).html",
        sample_rate: 22050,
        cases: {},
    };

    for (const { name, params } of cases) {
        const out = await runCase(name, params);
        reference.cases[name] = out;
        console.log(`[ok] ${name}: ${out.samples.length} samples, ratio=${out.ratio.toFixed(4)}, db=${out.db ? out.db.n : null}`);
    }

    reference.signals_provenance = {
        file: "signals.json",
        recipe_sha256: signals.recipe_sha256,
        signals_sha256: signals.sha256,
    };

    fs.mkdirSync(REF_DIR, { recursive: true });
    fs.writeFileSync(OUTPUT, JSON.stringify(reference));
    console.log(`wrote ${OUTPUT}`);
}

main().catch((err) => {
    console.error(err);
    process.exit(1);
});
