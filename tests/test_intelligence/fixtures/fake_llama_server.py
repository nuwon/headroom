#!/usr/bin/env python3
"""A tiny stand-in for llama.cpp's llama-server used by subprocess tests.

Implements --version / --help / --list-devices and serves /health,
/tokenize (parse_special) and /completion (n_probs top_logprobs). Logprobs
are deterministic functions of the prompt so decisions are reproducible.
Env knobs: FAKE_LLAMA_OOM=1 (exit with an OOM message unless -ngl 0),
FAKE_LLAMA_LOAD_S (seconds of 503 "Loading model" before ready),
FAKE_LLAMA_OLD=1 (help without --hf-repo).
"""

import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LETTERS = "ABCDEFGHIJKLMNOP"
args = sys.argv[1:]
if "--version" in args:
    print("version: 9999 (abc1234)\nbuilt with fake for x86_64", file=sys.stderr)
    sys.exit(0)
if "--help" in args:
    if os.environ.get("FAKE_LLAMA_OLD"):
        print("-m, --model FNAME\n-ngl, --n-gpu-layers N\n")
    else:
        print(
            "-hfr, --hf-repo <user>/<model>[:quant]\n-hff, --hf-file FILE\n"
            "-ngl, --gpu-layers, --n-gpu-layers N   max. number of layers to store in VRAM (default: auto)\n"
            "--list-devices\n-c, --ctx-size N\n--host HOST\n--port PORT\n"
        )
    sys.exit(0)
if "--list-devices" in args:
    print("Available devices:\n  CPU: Fake CPU (16000 MiB, 16000 MiB free)")
    sys.exit(0)


def arg(name, default=None):
    if name in args:
        return args[args.index(name) + 1]
    return default


port = int(arg("--port", "8080"))
host = arg("--host", "127.0.0.1")
ngl = arg("--n-gpu-layers", "auto")
if os.environ.get("FAKE_LLAMA_OOM") and ngl != "0":
    print(
        "ggml_backend_cuda_buffer_type_alloc_buffer: cudaMalloc failed: out of memory",
        file=sys.stderr,
    )
    sys.exit(1)
ready_at = time.monotonic() + float(os.environ.get("FAKE_LLAMA_LOAD_S", "0"))


def logprobs_for(prompt):
    start = prompt.find("<|im_start|>user\n")
    end = prompt.find("<|im_end|>", start)
    payload = json.loads(prompt[start + len("<|im_start|>user\n") : end])
    n = len(payload["options"])
    seed = sum(map(ord, payload["criterion"])) % 7
    out = []
    for i in range(min(n, 16)):
        out.append({"token": LETTERS[i], "logprob": -0.2 - abs(i - (seed % n)) * 0.9})
    out.append({"token": " the", "logprob": -9.0})
    return out


class H(BaseHTTPRequestHandler):
    def _send(self, status, payload):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            if time.monotonic() < ready_at:
                self._send(503, {"error": {"code": 503, "message": "Loading model"}})
            else:
                self._send(200, {"status": "ok"})
        else:
            self._send(404, {})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        if self.path == "/tokenize":
            assert body.get("parse_special") is True
            self._send(200, {"tokens": [ord(c) for c in body["content"]]})
        elif self.path == "/completion":
            prompt = "".join(chr(t) for t in body["prompt"])
            top = sorted(logprobs_for(prompt), key=lambda e: -e["logprob"])[
                : body.get("n_probs", 40)
            ]
            self._send(
                200,
                {
                    "completion_probabilities": [{"top_logprobs": top}],
                    "tokens_evaluated": len(body["prompt"]),
                    "timings": {"prompt_ms": 1.0, "predicted_ms": 0.1},
                },
            )
        else:
            self._send(404, {})

    def log_message(self, *a):
        pass


ThreadingHTTPServer((host, port), H).serve_forever()
