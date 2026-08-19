// worker.js
// Runs the AI model in a background thread so the page never freezes.
// Everything here happens on-device: no server, no API key, nothing sent anywhere.
//
// Powered by Hugging Face's transformers.js — loaded from a CDN, no build step needed.
// "+esm" tells jsDelivr to serve a ready-to-import browser module, which is the
// most reliable way to pull an npm package straight into a <script type="module">.
import {
  pipeline,
  TextStreamer,
  env,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers/+esm";

env.allowLocalModels = false;
env.useBrowserCache = true;
// Keep WASM threading modest — school/shared computers often have limited CPU headroom.
env.backends.onnx.wasm.numThreads = Math.max(
  1,
  Math.min(4, Math.floor((navigator.hardwareConcurrency || 4) / 2))
);

// Three real, open-weight models (Apache 2.0) that all support "thinking mode" —
// they reason in a <think>...</think> block before answering, which is what
// powers the visible "thinking time" in the UI.
const MODELS = {
  fast: { id: "onnx-community/Qwen3-0.6B-ONNX", label: "Qwen3 0.6B" },
  balanced: { id: "onnx-community/Qwen3-1.7B-ONNX", label: "Qwen3 1.7B" },
  quality: { id: "onnx-community/Qwen3-4B-ONNX", label: "Qwen3 4B" },
};

let generator = null;
let currentDevice = null;

class StreamHandler extends TextStreamer {
  constructor(tokenizer, onText) {
    super(tokenizer, { skip_prompt: true, skip_special_tokens: false });
    this.onText = onText;
  }
  on_finalized_text(text) {
    this.onText(text);
  }
}

async function loadModel(tier) {
  const cfg = MODELS[tier];
  if (!cfg) throw new Error("Unknown model: " + tier);

  const attempt = async (device) => {
    // fp16 quantization needs a GPU; plain int4 is the safe, broadly-supported
    // choice for CPU (WASM) fallback.
    const dtype = device === "webgpu" ? "q4f16" : "q4";
    generator = await pipeline("text-generation", cfg.id, {
      dtype,
      device,
      progress_callback: (p) => postMessage({ type: "progress", data: p }),
    });
    currentDevice = device;
  };

  const gpuAvailable = typeof navigator !== "undefined" && !!navigator.gpu;
  if (gpuAvailable) {
    try {
      await attempt("webgpu");
    } catch {
      await attempt("wasm"); // graceful fallback if WebGPU init fails on this machine
    }
  } else {
    await attempt("wasm");
  }

  postMessage({ type: "ready", device: currentDevice, label: cfg.label });
}

async function generate({ messages, maxNewTokens }) {
  if (!generator) throw new Error("Model isn't loaded yet.");

  let tokenCount = 0;
  const t0 = performance.now();
  let thinkEndElapsed = null;
  let sawClose = false;

  const streamer = new StreamHandler(generator.tokenizer, (chunk) => {
    tokenCount += 1;
    if (!sawClose && chunk.includes("</think>")) {
      sawClose = true;
      thinkEndElapsed = performance.now() - t0;
    }
    postMessage({ type: "chunk", data: chunk, elapsed: performance.now() - t0 });
  });

  const output = await generator(messages, {
    max_new_tokens: maxNewTokens || 1024,
    // Qwen3's own guidance: never use greedy decoding in thinking mode — it
    // causes repetition loops. These are their recommended sampling settings.
    do_sample: true,
    temperature: 0.6,
    top_p: 0.95,
    top_k: 20,
    streamer,
  });

  const totalTime = performance.now() - t0;

  postMessage({
    type: "done",
    fullText: output[0].generated_text.at(-1).content,
    stats: {
      totalTime,
      thinkTime: thinkEndElapsed,
      tokenCount,
      tokPerSec: tokenCount / (totalTime / 1000),
      device: currentDevice,
    },
  });
}

self.onmessage = async (e) => {
  const { type, payload } = e.data || {};
  try {
    if (type === "load" || type === "switch") {
      generator = null;
      await loadModel(payload.tier);
    } else if (type === "generate") {
      await generate(payload);
    }
  } catch (err) {
    postMessage({ type: "error", message: (err && err.message) || String(err) });
  }
};
