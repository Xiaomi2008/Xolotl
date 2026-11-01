#!/usr/bin/env python3
"""Strata's API server (~/AI/Strata/serve/server.py: OpenAI + Anthropic endpoints, streaming, sessions) in front of
the GLM-5.3-Flash engine (`glm_generate --serve`).  The server is Qwen3.8's; this shim swaps in GLM's pieces:

- tokenizer: the GGUF's BPE (tokenizer dir from export_tokenizer.py) with llama.cpp's CHATGLM4 pre-split
- chat template: the GGUF's (chat_template.jinja in the same dir; the server loads it from there)
- end of turn: <|user|> / <|observation|> / <|endoftext|> instead of <|im_end|>
- tool calls: GLM's `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>...</tool_call>` (values: JSON when
  the template wrote JSON, i.e. the schema says the parameter is not a string)

  serve_glm.py --engine strata --config strata-glm.json --port 8140
"""
import json
import os
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
# xolotl: the server and tokenizer are THIS tree's (they used to be ~/AI/Strata's, a separate upstream checkout at 0.1.39)
STRATA = HERE.parent.parent.parent
sys.path.insert(0, str(STRATA))
sys.path.insert(0, str(STRATA / "tools"))
sys.path.insert(0, str(HERE))

import strata_tokenizer as ST  # noqa: E402
import glm_tok  # noqa: E402
from serve import frontend as F  # noqa: E402
from serve import server as S  # noqa: E402

# ---- tokenizer: the server builds ST.Tokenizer(tokens, merges, types); give it GLM's pre-split
_Tok = ST.Tokenizer


# xolotl: this tree's tokenizer picks its pre-split from PRE_PATTERNS by name (the old one read a single self._re,
# which this class overwrote); register llama.cpp's CHATGLM4 split there
ST.PRE_PATTERNS.setdefault("glm4", [glm_tok.GLM4_PATTERN])


class GlmTokenizer(_Tok):
    def __init__(self, tokens, merges, types=None, pre="glm4", special_ids=None):
        super().__init__(tokens, merges, types, pre="glm4", special_ids=special_ids)


ST.Tokenizer = GlmTokenizer

# ---- end of turn
S.IM_END = "<|user|>"
_svc_init = S.Service.__init__


def _svc_init_glm(self, engine, tokenizer, template, *a, **k):
    _svc_init(self, engine, tokenizer, template, *a, **k)
    for t in ("<|observation|>", "<|endoftext|>", "<|user|>"):
        self.stop_ids |= set(tokenizer.encode(t, parse_special=True))


S.Service.__init__ = _svc_init_glm

# ---- tool calls in GLM's format
ARG_K, ARG_K_END, ARG_V, ARG_V_END = "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>"
_schemas = threading.local()


def glm_call_end(text: str) -> int:
    """The </tool_call> that closes the call: the first one outside an <arg_value> (a value may contain the tag)."""
    pos = 0
    while True:
        e = text.find(F.CALL_END, pos)
        if e < 0:
            return -1
        seg = text[:e]
        if seg.count(ARG_V) <= seg.count(ARG_V_END):
            return e
        pos = e + 1


def glm_parse_tool_call(body: str, schema=None):
    name, rest = body, ""
    if ARG_K in body:
        name, rest = body[:body.index(ARG_K)], body[body.index(ARG_K):]
    name = name.strip()
    if not name:
        raise ValueError("malformed tool call: " + body[:80])
    schema = schema or (getattr(_schemas, "v", None) or {}).get(name)
    props = ((schema or {}).get("parameters") or {}).get("properties") or {}
    args = {}
    while ARG_K in rest:
        rest = rest[rest.index(ARG_K) + len(ARG_K):]
        if ARG_K_END not in rest:
            break
        key = rest[:rest.index(ARG_K_END)].strip()
        rest = rest[rest.index(ARG_K_END) + len(ARG_K_END):]
        if ARG_V not in rest:
            args[key] = ""
            break
        rest = rest[rest.index(ARG_V) + len(ARG_V):]
        end, at = -1, rest.find(ARG_V_END)
        while at >= 0:   # the </arg_value> followed by the next <arg_key> or the end (a value may contain the tag)
            after = rest[at + len(ARG_V_END):].lstrip()
            if not after or after.startswith(ARG_K):
                end = at
                break
            at = rest.find(ARG_V_END, at + 1)
        value = rest[:end] if end >= 0 else rest
        rest = rest[end + len(ARG_V_END):] if end >= 0 else ""
        if (props.get(key) or {}).get("type") == "string":
            args[key] = value
        else:
            try:
                args[key] = json.loads(value)
            except ValueError:
                args[key] = value
    return F.ToolCall(name=name, arguments=args)


# 2026-11-01: since the upstream merge the parser only reads Qwen's `<tool_call><function=NAME><parameter=K>V` form (the call
# regex is fixed), so the old patches (call_end / parse_tool_call / FUNC_START = "") left every GLM call as answer text.
# GlmOutputParser (below) holds a call from <tool_call> to its </tool_call>, parses it GLM's way and hands the parser the
# same call in the Qwen form - the approach upstream takes for DeepSeek's DSML (serve/dsml.py).


def _partial(text: str, tag: str) -> int:
    """Length of the longest end of `text` that is the start of `tag` (held: the tag may still be arriving)."""
    for n in range(min(len(tag) - 1, len(text)), 0, -1):
        if tag.startswith(text[-n:]):
            return n
    return 0


def qwen_call_text(call) -> str:
    parts = [f"{F.CALL_START}\n<function={call.name}>\n"]
    for k, v in (call.arguments or {}).items():
        parts.append(f"<parameter={k}>\n{v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)}\n</parameter>\n")
    parts.append(f"</function>\n{F.CALL_END}")
    return "".join(parts)
_op_init = F.OutputParser.__init__


def _op_init_glm(self, *a, **k):
    _op_init(self, *a, **k)
    _schemas.v = self.schemas          # parse_tool_call gets the name GLM's way; the schema comes from here
    self.stream_tools = False          # the streaming scanner reads Qwen's <function=...> body


F.OutputParser.__init__ = _op_init_glm
S.OutputParser = F.OutputParser


class GlmOutputParser(F.OutputParser):
    """GLM calls -> the Qwen form before the parser (see qwen_call_text)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.glm_buf = ""

    def _glm_translate(self, final: bool) -> str:
        out, b = "", self.glm_buf
        while True:
            i = b.find(F.CALL_START)
            if i < 0:
                keep = 0 if final else _partial(b, F.CALL_START)
                out += b[:len(b) - keep]
                b = b[len(b) - keep:]
                break
            out += b[:i]
            rest = b[i + len(F.CALL_START):]
            e = glm_call_end(rest)
            if e < 0:                          # the call is still arriving (or never closed: passed on as it is)
                if final:
                    out += b[i:]
                    b = ""
                else:
                    b = b[i:]
                break
            try:
                out += qwen_call_text(glm_parse_tool_call(rest[:e]))
            except ValueError:
                out += b[i:i + len(F.CALL_START) + e + len(F.CALL_END)]
            b = rest[e + len(F.CALL_END):]
        self.glm_buf = b
        return out

    def feed(self, delta: str):
        self.glm_buf += delta
        text = self._glm_translate(False)
        return super().feed(text) if text else []

    def finish(self, *a, **k):
        text = self._glm_translate(True)
        out = super().feed(text) if text else []
        return out + super().finish(*a, **k)


S.OutputParser = GlmOutputParser       # Service.parser_cls (read at Service init)

if __name__ == "__main__":
    for i, v in enumerate(sys.argv[:-1]):   # paths are the caller's; the server runs from the repo root
        if v in ("--config", "--tokenizer", "--mcp-config"):
            sys.argv[i + 1] = os.path.abspath(sys.argv[i + 1])
    os.chdir(str(STRATA))
    sys.exit(S.main())
