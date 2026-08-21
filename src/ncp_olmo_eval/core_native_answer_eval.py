#!/usr/bin/env python3
"""Official-answer scoring contracts used by the Core88 evaluator.

The GSM and Minerva/MATH behavior in this module is pinned to
allenai/olmo-eval commit ``f8816eea36563f27b4a9dd2533d68d34f3c67d3f``.
The math extraction and equivalence helpers are adapted from
``src/olmo_eval/evals/extract/math.py`` at that commit. The original source
SHA-256 is ``53b5d0a744b31120c9df3a5078a3a3d0dd7dd846d44925deae645e39eb3addc4``.

DeepMind Mathematics QA and BBH follow the fixed OLMo paper task recipes that
were used to render the immutable Core88 inputs. They are kept separate from
the current olmo-eval contracts because those two legacy tasks are not present
in olmo-eval at the pinned commit.
"""

from __future__ import annotations

import importlib.metadata
import logging
import re
import signal
from functools import lru_cache
from typing import Any

log = logging.getLogger(__name__)

OFFICIAL_OLMO_EVAL_COMMIT = "f8816eea36563f27b4a9dd2533d68d34f3c67d3f"
OFFICIAL_MATH_SOURCE_SHA256 = "53b5d0a744b31120c9df3a5078a3a3d0dd7dd846d44925deae645e39eb3addc4"
OFFICIAL_GSM8K_SOURCE_SHA256 = "b7175e309460b36578291648ff1262c66b60d314892a0ed4e8e8585efdd90228"
OFFICIAL_GSM_SYMBOLIC_SOURCE_SHA256 = (
    "50be5423cab388bb4a8d48849022a820e27208af9fe52dabeb0ed029cd82fe99"
)
OFFICIAL_ANTLR_VERSION_PREFIX = "4.11."
CORE88_GENERATED_UTILS_SHA256 = (
    "94cbc08b6a8cf0a5c293b135118966db1af527f7c95200b7f697a34a6648015b"
)
CORE88_DEEPMIND_TASK_SHA256 = (
    "18625e2ced883846cd7449b6c9156369a5934aeb71d4b53bafc163ebc397b86a"
)
CORE88_BBH_FILTER_RECIPE_SHA256 = (
    "c434c16a3ad17e3f2310e1f77d5bbebb971c3f3c532445bc232f6c1c4c95e9fd"
)

_NUMBER_RE = re.compile(r"[-+]?\d*\.\d+|[-+]?\d+")
_COMMA_IN_NUMBER_RE = re.compile(r"(\d),(\d)")
_BBH_ANSWER_RE = re.compile(
    r"(?i)(?:the answer is|answer:)\s*([^.\n]+)"
)

_SUBSTITUTIONS = (
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
)

_REMOVED_EXPRESSIONS = (
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "ft",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
)


def extract_gsm_answer(text: str) -> str | None:
    """Extract the last numeric answer exactly as official OLMo-Eval GSM tasks do."""

    output = _COMMA_IN_NUMBER_RE.sub(r"\1\2", str(text))
    numbers = _NUMBER_RE.findall(output)
    return numbers[-1] if numbers else None


def clean_gsm_gold(text: str) -> str:
    """Clean a GSM gold answer using the official last-number contract."""

    output = _COMMA_IN_NUMBER_RE.sub(r"\1\2", str(text))
    numbers = _NUMBER_RE.findall(output)
    return numbers[-1] if numbers else str(text)


def score_gsm_answer(completion: str, gold: str) -> dict[str, Any]:
    """Score one GSM8K/GSM-Symbolic completion with the official exact-match contract."""

    extracted_prediction = extract_gsm_answer(completion)
    extracted_gold = clean_gsm_gold(gold)
    return {
        "normalized_prediction": extracted_prediction,
        "normalized_gold": extracted_gold,
        "extracted_predictions": (
            [] if extracted_prediction is None else [extracted_prediction]
        ),
        "extracted_golds": [extracted_gold],
        "primary_correct": extracted_prediction == extracted_gold,
        "score_status": "SCORED",
        "answer_scorer": "olmo_eval_gsm_last_number_exact_match",
        "olmo_eval_commit": OFFICIAL_OLMO_EVAL_COMMIT,
        "official_source_sha256s": [
            OFFICIAL_GSM8K_SOURCE_SHA256,
            OFFICIAL_GSM_SYMBOLIC_SOURCE_SHA256,
        ],
    }


def _extract_core88_boxed(text: str) -> str:
    stripped = str(text).strip()
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", stripped)
    if boxed:
        return boxed[-1].strip()
    if "####" in stripped:
        return stripped.rsplit("####", 1)[-1].strip()
    return stripped


def normalize_core88_exact_match(text: str) -> str:
    """Match the fixed OLMo paper ``utils._normalize_em_text`` helper."""

    normalized = str(text).strip()
    boxed = _extract_core88_boxed(normalized)
    if boxed != normalized:
        normalized = boxed
    else:
        matches = re.findall(
            r"(?is)(?:####|final answer:|the final answer is|the answer is)"
            r"\s*([^.\n]+)",
            normalized,
        )
        if matches:
            normalized = matches[-1]
    normalized = normalized.strip().strip(".。")
    normalized = normalized.replace("$", "").replace(",", "")
    normalized = re.sub(r"\\left|\\right", "", normalized)
    normalized = re.sub(r"\s+", "", normalized)
    return normalized.lower()


def score_deepmind_answer(completion: str, gold: str) -> dict[str, Any]:
    """Score the fixed OLMo paper DeepMind Mathematics QA task."""

    prediction = normalize_core88_exact_match(completion.strip())
    normalized_gold = normalize_core88_exact_match(gold)
    return {
        "normalized_prediction": prediction,
        "normalized_gold": normalized_gold,
        "primary_correct": prediction == normalized_gold,
        "score_status": "SCORED",
        "answer_scorer": "olmo_paper_deepmind_pass1_normalized_exact_match",
        "core88_generated_utils_sha256": CORE88_GENERATED_UTILS_SHA256,
        "core88_task_recipe_sha256": CORE88_DEEPMIND_TASK_SHA256,
        "official_source_sha256s": [
            CORE88_GENERATED_UTILS_SHA256,
            CORE88_DEEPMIND_TASK_SHA256,
        ],
    }


def score_bbh_answer(completion: str, gold: str) -> dict[str, Any]:
    """Apply the fixed BBH regex filter followed by strict exact match."""

    matches = _BBH_ANSWER_RE.findall(str(completion))
    prediction = matches[-1].strip() if matches else "[invalid]"
    expected = str(gold)
    return {
        "normalized_prediction": prediction,
        "normalized_gold": expected,
        "primary_correct": prediction == expected,
        "score_status": "SCORED",
        "answer_scorer": "olmo_paper_bbh_regex_then_strict_exact_match",
        "core88_task_recipe_sha256": CORE88_BBH_FILTER_RECIPE_SHA256,
        "official_source_sha256s": [CORE88_BBH_FILTER_RECIPE_SHA256],
    }


def _last_boxed_only_string(text: str) -> str | None:
    index = text.rfind("\\boxed")
    if "\\boxed " in text:
        return "\\boxed " + text.split("\\boxed ")[-1].split("$")[0]
    if index < 0:
        index = text.rfind("\\fbox")
        if index < 0:
            return None

    right_brace_index = None
    open_braces = 0
    for cursor in range(index, len(text)):
        if text[cursor] == "{":
            open_braces += 1
        if text[cursor] == "}":
            open_braces -= 1
            if open_braces == 0:
                right_brace_index = cursor
                break
    return (
        None
        if right_brace_index is None
        else text[index : right_brace_index + 1]
    )


def _remove_boxed(text: str) -> str:
    if "\\boxed " in text:
        prefix = "\\boxed "
        if not text.startswith(prefix):
            raise AssertionError(f"invalid boxed answer: {text!r}")
        return text[len(prefix) :]

    prefix = "\\boxed{"
    if not text.startswith(prefix) or not text.endswith("}"):
        raise AssertionError(f"invalid boxed answer: {text!r}")
    return text[len(prefix) : -1]


def _get_unnormalized_answer(text: str) -> str:
    invalid_answer = "[invalidanswer]"
    end_sequence = "I hope it is correct."
    match = re.search(
        r"Final Answer: The final answer is(.*?). I hope it is correct.",
        str(text) + end_sequence,
    )
    return match.group(1).strip() if match else invalid_answer


def _normalize_final_answer(final_answer: str) -> str:
    final_answer = str(final_answer).split("=")[-1]

    for before, after in _SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expression in _REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expression, "")

    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")
    return final_answer


def extract_math_answers(text: str) -> list[str]:
    """Extract every official Minerva/MATH answer candidate from a completion."""

    result = str(text)
    boxed_answer = _last_boxed_only_string(result)
    if boxed_answer is not None:
        try:
            boxed_answer = _remove_boxed(boxed_answer)
        except AssertionError:
            boxed_answer = None

    answers = []
    minerva_answer = _normalize_final_answer(_get_unnormalized_answer(result))
    if minerva_answer and minerva_answer != "[invalidanswer]":
        answers.append(minerva_answer)
    if boxed_answer is not None:
        answers.append(_normalize_final_answer(boxed_answer))
    if not answers:
        dollar_indices = [match.start() for match in re.finditer("\\$", result)]
        if len(dollar_indices) > 1:
            answers.append(
                _normalize_final_answer(
                    result[dollar_indices[-2] + 1 : dollar_indices[-1]]
                )
            )
    if not answers:
        answers.append(_normalize_final_answer(result))
    return answers


def _fix_fracs(text: str) -> str:
    substrings = text.split("\\frac")
    output = substrings[0]
    for substring in substrings[1:]:
        output += "\\frac"
        if substring[0] == "{":
            output += substring
            continue
        if len(substring) < 2:
            return text
        numerator = substring[0]
        denominator = substring[1]
        remainder = substring[2:]
        if denominator != "{":
            output += "{" + numerator + "}{" + denominator + "}" + remainder
        else:
            output += "{" + numerator + "}" + denominator + remainder
    return output


def _fix_a_slash_b(text: str) -> str:
    if len(text.split("/")) != 2:
        return text
    numerator_text, denominator_text = text.split("/")
    try:
        numerator = int(numerator_text)
        denominator = int(denominator_text)
    except ValueError:
        return text
    if text != f"{numerator}/{denominator}":
        return text
    return "\\frac{" + str(numerator) + "}{" + str(denominator) + "}"


def _remove_right_units(text: str) -> str:
    if "\\text{ " not in text:
        return text
    pieces = text.split("\\text{ ")
    if len(pieces) != 2:
        raise AssertionError(f"unexpected unit expression: {text!r}")
    return pieces[0]


def _fix_sqrt(text: str) -> str:
    if "\\sqrt" not in text:
        return text
    pieces = text.split("\\sqrt")
    output = pieces[0]
    for piece in pieces[1:]:
        if piece[0] != "{":
            output += "\\sqrt{" + piece[0] + "}" + piece[1:]
        else:
            output += "\\sqrt" + piece
    return output


def _strip_math_string(text: str) -> str:
    text = text.replace("\n", "")
    text = text.replace("\\!", "")
    text = text.replace("\\\\", "\\")
    text = text.replace("tfrac", "frac")
    text = text.replace("dfrac", "frac")
    text = text.replace("\\left", "")
    text = text.replace("\\right", "")
    text = text.replace("^{\\circ}", "")
    text = text.replace("^\\circ", "")
    text = text.replace("\\$", "")
    text = _remove_right_units(text)
    text = text.replace("\\%", "")
    text = text.replace(r"\%", "")
    text = text.replace(" .", " 0.")
    text = text.replace("{.", "{0.")
    if not text:
        return text
    if text[0] == ".":
        text = "0" + text
    if len(text.split("=")) == 2 and len(text.split("=")[0]) <= 2:
        text = text.split("=")[1]
    text = _fix_sqrt(text)
    text = text.replace(" ", "")
    text = _fix_fracs(text)
    if text == "0.5":
        text = "\\frac{1}{2}"
    return _fix_a_slash_b(text)


def _hendrycks_is_equiv(first: str | None, second: str | None) -> bool:
    if first is None and second is None:
        return True
    if first is None or second is None:
        return False
    try:
        return _strip_math_string(first) == _strip_math_string(second)
    except Exception:
        return first == second


class _AlarmTimeout:
    def __init__(self, seconds: int = 5) -> None:
        self.seconds = seconds

    @staticmethod
    def _handle_timeout(signum: int, frame: Any) -> None:
        del signum, frame
        raise TimeoutError("official Minerva symbolic comparison timed out")

    def __enter__(self) -> None:
        signal.signal(signal.SIGALRM, self._handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        del exc_type, exc_value, traceback
        signal.alarm(0)


@lru_cache(maxsize=1)
def require_official_math_runtime() -> dict[str, str]:
    """Validate the exact symbolic runtime required by official Minerva scoring."""

    sympy_version = importlib.metadata.version("sympy")
    antlr_version = importlib.metadata.version("antlr4-python3-runtime")
    if not antlr_version.startswith(OFFICIAL_ANTLR_VERSION_PREFIX):
        raise RuntimeError(
            "official Minerva scorer requires antlr4-python3-runtime>=4.11,<4.12, "
            f"got {antlr_version}"
        )
    from sympy.parsing.latex import parse_latex

    parsed = parse_latex(r"\frac{1}{2}")
    if str(parsed) != "1/2":
        raise RuntimeError(f"official Minerva parser smoke returned {parsed!r}")
    return {
        "sympy": sympy_version,
        "antlr4-python3-runtime": antlr_version,
        "olmo_eval_commit": OFFICIAL_OLMO_EVAL_COMMIT,
        "math_source_sha256": OFFICIAL_MATH_SOURCE_SHA256,
    }


def _minerva_is_equiv(first: str, second: str) -> bool:
    try:
        import sympy
        import sympy.parsing.latex.errors
        from sympy.parsing.latex import parse_latex

        with _AlarmTimeout(seconds=5):
            try:
                parsed_first = parse_latex(first)
                parsed_second = parse_latex(second)
            except (
                sympy.parsing.latex.errors.LaTeXParsingError,
                sympy.SympifyError,
                TypeError,
            ):
                return False
            try:
                difference = parsed_first - parsed_second
            except TypeError:
                return False
            try:
                return sympy.simplify(difference) == 0
            except ValueError:
                return False
    except (ImportError, TimeoutError):
        return False
    except Exception as error:
        log.debug("official Minerva comparison failed: %s", error)
        return False


def math_is_equiv(first: str | None, second: str | None) -> bool:
    """Compare two answers with official Minerva then Hendrycks equivalence."""

    if first is None or second is None:
        return first is None and second is None
    if _minerva_is_equiv(first, second):
        return True
    return _hendrycks_is_equiv(first, second)


def score_math_answer(
    completion: str,
    gold: str,
    *,
    require_official_runtime: bool,
) -> dict[str, Any]:
    """Score one Minerva/MATH completion against every extracted gold answer."""

    runtime = (
        require_official_math_runtime() if require_official_runtime else None
    )
    predictions = extract_math_answers(completion)
    golds = extract_math_answers(gold)
    correct = any(
        math_is_equiv(prediction.strip(), target.strip())
        for prediction in predictions
        for target in golds
    )
    return {
        "normalized_prediction": predictions[0] if predictions else None,
        "normalized_gold": golds[0] if golds else None,
        "extracted_predictions": predictions,
        "extracted_golds": golds,
        "primary_correct": correct,
        "score_status": "SCORED",
        "answer_scorer": "olmo_eval_minerva_math_flex",
        "olmo_eval_commit": OFFICIAL_OLMO_EVAL_COMMIT,
        "official_math_runtime": runtime,
        "official_source_sha256s": [OFFICIAL_MATH_SOURCE_SHA256],
    }
