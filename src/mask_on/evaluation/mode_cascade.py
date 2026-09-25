"""Parse review verdicts, corrections, and answer-format status."""

import re
from decimal import Decimal


def read_correction(text):
    """Parse CORRECT or INCORRECT followed by a numeric #### answer."""
    text = (text or '').strip()
    if text == 'CORRECT':
        return 'accept', 'NULL'
    number = r'[+-]?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?'
    match = re.fullmatch(r'INCORRECT\r?\n#### (' + number + r')', text)
    if match:
        return 'reject', match[1].replace(',', '')
    return None, 'NULL'


def parse_semantic_review(text, draft_answer):
    """Parse verdicts, matching answer echoes, and explicit numeric corrections.

    Return semantic interpretation and format compliance as separate fields.
    """
    text = (text or '').replace('\r\n', '\n').strip()
    number = r'[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?'
    result = dict(verdict=None, correction='NULL', canonical_format=False,
                  interpretation='unparseable', contradictory=False)
    if text == 'CORRECT':
        return dict(result, verdict='accept', canonical_format=True,
                    interpretation='canonical')
    # Match a verdict separated from its numeric answer.
    echo = re.fullmatch(r'CORRECT(?:\s+(?:####[ \t]*)?|####[ \t]*)(' + number + r')', text)
    if echo:
        if not re.fullmatch(number, str(draft_answer)):
            return dict(result, interpretation='missing_numeric_draft')
        if Decimal(echo[1].replace(',', '')) != Decimal(str(draft_answer).replace(',', '')):
            return dict(result, interpretation='contradictory_echo', contradictory=True)
        return dict(result, verdict='accept', interpretation='matching_numeric_echo')
    header = re.fullmatch(r'INCORRECT(?:\s+|(?=####))(.*)', text, re.S)
    if not header:
        return result
    body = header[1].strip()
    if re.search(r'\b(?:CORRECT|INCORRECT)\b', body):
        return dict(result, interpretation='multiple_verdicts', contradictory=True)
    if '####' in body:
        if body.count('####') != 1:
            return dict(result, interpretation='multiple_final_markers')
        match = re.fullmatch(r'(.*?)####[ \t]*(' + number + r')', body, re.S)
        if not match:
            return dict(result, interpretation='nonterminal_or_nonnumeric_final')
        canonical = bool(re.fullmatch(r'INCORRECT\n(?:.+\n)?#### ' + number, text, re.S))
        return dict(result, verdict='reject', correction=match[2].replace(',', ''),
                    canonical_format=canonical, interpretation='explicit_final_correction')
    if re.fullmatch(number, body):
        return dict(result, verdict='reject', correction=body.replace(',', ''),
                    interpretation='unmarked_numeric_correction')
    return dict(result, interpretation='missing_explicit_final')


def parse_explicit_review(text, draft_answer, *, allow_consistent_markers=False):
    """Parse a verdict with a terminal #### or LaTeX-boxed numeric answer.

    Check accepted answer echoes against the saved draft answer.
    """
    text = (text or '').replace('\r\n', '\n').strip()
    result = dict(verdict=None, correction='NULL', canonical_format=False,
                  interpretation='unparseable', contradictory=False)
    header = re.fullmatch(r'(CORRECT|INCORRECT)(?:\s+|(?=####|\\boxed))(.*)', text, re.S)
    if not header:
        return parse_semantic_review(text, draft_answer)
    verdict, body = header[1], header[2].strip()
    if re.search(r'\b(?:CORRECT|INCORRECT)\b', body):
        return dict(result, interpretation='multiple_verdicts', contradictory=True)
    markers = body.count('####') + len(re.findall(r'\\boxed\s*\{', body))
    if markers == 0:
        return parse_semantic_review(text, draft_answer)
    if markers != 1 and not allow_consistent_markers:
        return dict(result, interpretation='multiple_final_markers')
    number = r'[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?'
    if allow_consistent_markers:
        # Check that every explicit answer marker parses to the same value.
        values = []
        for marker in re.finditer(r'####|\\boxed\s*\{', body):
            suffix = body[marker.start():]
            numeric = re.match(r'####[ \t]*(' + number + r')(?=$|\s|[.$]|\\[\])])'
                               r'|\\boxed\s*\{\s*(' + number + r')\s*\}', suffix)
            if not numeric:
                return dict(result, interpretation='nonnumeric_explicit_marker')
            values.append(Decimal((numeric[1] or numeric[2]).replace(',', '')))
        if len(set(values)) != 1:
            return dict(result, interpretation='conflicting_final_markers', contradictory=True)
    final = re.fullmatch(
        r'(.*?)(?:####[ \t]*(' + number + r')|\\boxed\s*\{\s*('
        + number + r')\s*\})(?:\s|[.$]|\\[\])])*', body, re.S)
    if not final:
        return dict(result, interpretation='nonterminal_or_nonnumeric_final')
    value = (final[2] or final[3]).replace(',', '')
    if verdict == 'CORRECT':
        if not re.fullmatch(number, str(draft_answer)):
            return dict(result, interpretation='missing_numeric_draft')
        if Decimal(value) != Decimal(str(draft_answer).replace(',', '')):
            return dict(result, interpretation='contradictory_echo', contradictory=True)
        return dict(result, verdict='accept', interpretation='matching_explicit_echo')
    canonical = parse_semantic_review(text, draft_answer)['canonical_format']
    return dict(result, verdict='reject', correction=value, canonical_format=canonical,
                interpretation='explicit_boxed_correction' if final[3]
                else 'explicit_final_correction')


def parse_review(text, draft_answer, policy='strict_v1'):
    """Dispatch to the selected review parser and return verdict and format status."""
    if policy not in ('strict_v1', 'matching_echo_v1', 'verdict_number_v2', 'reasoned_correction_v1', 'review_semantic_v1', 'review_semantic_v2', 'review_semantic_v3'):
        raise ValueError('unknown review parser')
    if policy == 'review_semantic_v1':
        return parse_semantic_review(text, draft_answer)
    if policy == 'review_semantic_v2':
        return parse_explicit_review(text, draft_answer)
    if policy == 'review_semantic_v3':
        return parse_explicit_review(text, draft_answer, allow_consistent_markers=True)
    if policy == 'reasoned_correction_v1':
        text = (text or '').strip()
        number = r'[+-]?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?'
        result = dict(verdict=None, correction='NULL', canonical_format=False,
                      interpretation='unparseable', contradictory=False)
        if text == 'CORRECT':
            return dict(result, verdict='accept', canonical_format=True, interpretation='canonical')
        match = re.fullmatch(r'INCORRECT\r?\n(.+)\r?\n####[ \t]+(' + number + r')', text, re.S)
        if not match:
            return result
        reasoning = match[1].strip()
        # Reject repeated verdicts or answer markers in the explanation.
        if (not reasoning or '####' in reasoning
                or re.search(r'(?m)^\s*(?:CORRECT|INCORRECT)\s*$', reasoning)):
            return result
        return dict(result, verdict='reject', correction=match[2].replace(',', ''),
                    canonical_format=True, interpretation='reasoned_correction')
    verdict, answer = read_correction(text)
    result = dict(verdict=verdict, correction=answer,
                  canonical_format=verdict is not None,
                  interpretation='canonical' if verdict is not None else 'unparseable',
                  contradictory=False)
    if verdict is not None or policy == 'strict_v1':
        return result
    number = r'[+-]?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?'
    if policy == 'verdict_number_v2':
        correction = re.fullmatch(r'INCORRECT\r?\n(?:####[ \t]*)?(' + number + r')', (text or '').strip())
        if correction:
            return dict(result, verdict='reject', correction=correction[1].replace(',',''),
                        interpretation='unmarked_numeric_correction')
    delimiter = r'(?:####[ \t]*)?' if policy == 'verdict_number_v2' else r'(?:#### )?'
    match = re.fullmatch(r'CORRECT\r?\n' + delimiter + '(' + number + r')', (text or '').strip())
    if not match or not re.fullmatch(number, str(draft_answer)):
        return result
    echoed = Decimal(match[1].replace(',', ''))
    draft = Decimal(str(draft_answer).replace(',', ''))
    if echoed != draft:
        return dict(result, interpretation='contradictory_echo', contradictory=True)
    return dict(result, verdict='accept', interpretation='matching_numeric_echo')
