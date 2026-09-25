import builtins
import importlib.metadata
import random
import sys
import types
from functools import lru_cache
from pathlib import Path
from mask_on.artifacts import file_sha as sha
from .common import scorer_receipt
VERSION = 'ifeval-explicit-symbol-v1'
LETTER_ID = 'keywords:letter_frequency'


class NoRandomDefaults:
    def __getattr__(self, name):
        raise ValueError('unregistered IFEval random default: ' + name)


@lru_cache(maxsize=1)
def corrected_scorer():
    folder = Path(importlib.metadata.distribution('lm_eval').locate_file('lm_eval/tasks/ifeval'))
    ns = types.ModuleType('_mask_on_ifeval_explicit_symbol_v1')
    original_import = builtins.__import__

    def isolated_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == 'lm_eval.tasks.ifeval':
            return ns
        return original_import(name, globals, locals, fromlist, level)

    for name in ('instructions_util', 'instructions', 'instructions_registry', 'utils'):
        module = types.ModuleType(ns.__name__ + '.' + name)
        path = folder / (name + '.py')
        module.__file__ = str(path)
        module.__dict__['__builtins__'] = dict(vars(builtins), __import__=isolated_import)
        sys.modules[module.__name__] = module
        exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
        setattr(ns, name, module)

    parent = ns.instructions.LetterFrequencyChecker

    class ExplicitSymbolFrequencyChecker(parent):
        def build_description(self, *, letter=None, let_frequency=None, let_relation=None):
            if isinstance(letter, str) and len(letter) == 1 and not letter.isascii():
                raise ValueError('non-ASCII letter semantics are not registered')
            symbol = isinstance(letter, str) and len(letter) == 1 and not letter.isalpha()
            # Validate the supplied count and relation without random defaults.
            description = super().build_description(
                letter='a' if symbol else letter, let_frequency=let_frequency,
                let_relation=let_relation)
            if symbol:
                self._letter = letter
                return self._description_pattern.format(
                    letter=letter, let_frequency=self._frequency, let_relation=self._comparison_relation)
            return description

    assert ns.instructions_registry.INSTRUCTION_DICT[LETTER_ID] is parent
    ns.instructions_registry.INSTRUCTION_DICT = dict(ns.instructions_registry.INSTRUCTION_DICT)
    ns.instructions_registry.INSTRUCTION_DICT[LETTER_ID] = ExplicitSymbolFrequencyChecker
    ns.instructions.random = NoRandomDefaults()
    ns.instructions_util.random = NoRandomDefaults()
    return ns.utils


def correction_receipt():
    import nltk
    from langdetect import detector_factory
    punkt = Path(str(nltk.data.find('tokenizers/punkt_tab/english')))
    profiles = Path(detector_factory.__file__).parent / 'profiles'
    return dict(version=VERSION, upstream=scorer_receipt(),
                correction_sha256=sha(__file__),
                assets={str(p.relative_to(base)): sha(p) for base in (punkt, profiles)
                        for p in sorted(base.rglob('*')) if p.is_file()},
                semantics='literal supplied ASCII symbol; unchanged strict/loose and other checkers',
                random_defaults='forbidden', language_detector_seed=0,
                python=sys.version.split()[0])


def score_one(record, text):
    from langdetect import DetectorFactory
    seed = DetectorFactory.seed
    state = random.getstate()
    try:
        DetectorFactory.seed = 0
        result = corrected_scorer().process_results(record, [text])
    finally:
        DetectorFactory.seed = seed
    if random.getstate() != state:
        random.setstate(state)
        raise ValueError('IFEval scorer consumed global RNG')
    return dict(score=float(result['prompt_level_strict_acc']), **result)
