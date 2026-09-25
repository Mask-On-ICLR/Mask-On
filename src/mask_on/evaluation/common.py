import importlib
import importlib.metadata
import re
from pathlib import Path
from functools import lru_cache
from mask_on.artifacts import file_sha, content_sha
from .gsm8k_metrics import equal, postprocess, reference

def _official_scorer_installed():
    """Return whether the IFEval scorer and language detector can be imported."""
    try:
        official_metric_module('ifeval')
        importlib.import_module('langdetect')
    except Exception:                                                     # noqa: BLE001
        return False
    return True


@lru_cache(maxsize=None)
def official_scorer_installed():
    """Cache the IFEval scorer availability check for this process."""
    return _official_scorer_installed()


@lru_cache(maxsize=None)
def official_metric_module(task):
    """Load task scorer modules into a private namespace with task-local imports."""
    import builtins
    import sys
    import types
    if task not in ('ifeval','minerva_math'):raise ValueError('unregistered scorer')
    folder=Path(importlib.metadata.distribution('lm_eval').locate_file('lm_eval/tasks'))/task
    namespace=types.ModuleType('_mask_on_metric_'+task)
    names=('instructions_util','instructions','instructions_registry','utils') if task=='ifeval' else ('utils',)
    original_import=builtins.__import__
    def metric_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name=='lm_eval.tasks.ifeval':return namespace
        return original_import(name,globals,locals,fromlist,level)
    for name in names:
        module=types.ModuleType(namespace.__name__+'.'+name)
        path=folder/(name+'.py');module.__file__=str(path)
        module.__dict__['__builtins__']=dict(vars(builtins),__import__=metric_import)
        sys.modules[module.__name__]=module
        exec(compile(path.read_bytes(),str(path),'exec'),module.__dict__)
        setattr(namespace,name,module)
    return namespace.utils


def choice_prompt(question, choices):
    if not 2 <= len(choices) <= 5:
        raise ValueError('unsupported number of choices')
    return question.strip()+'\n'+'\n'.join(f'{chr(65+i)}. {v}' for i,v in enumerate(choices))+\
        '\nAnswer with only the letter of the correct option.'


def mbpp_prompt(record):
    return 'Write a Python function for the following task. Return only Python code.\n'+record['text']+\
        '\nTests:\n'+'\n'.join(record['test_list'])


def prepare_records(task, rows, demos=(), shots=None):
    records=[]
    for index, row in enumerate(rows):
        messages=[]
        if task=='gsm8k':
            if shots:
                if len(demos)!=shots: raise ValueError('GSM8K demonstration count mismatch')
                for d in demos:
                    messages.extend([dict(role='user',content=d['question']),dict(role='assistant',content=d['answer'])])
            prompt=row['question']+'\nSolve step by step. End with "#### <answer>".'
            target=reference(row['answer'])
        elif task=='arc_challenge':
            labels=row['choices']['label']
            target=chr(65+labels.index(str(row['answerKey'])))
            prompt=choice_prompt(row['question'],row['choices']['text'])
        elif task=='mmlu':
            subject_demos=[d for d in demos if d['subject']==row['subject']]
            expected=5 if shots is None else shots
            subject_demos=subject_demos[:expected]
            if len(subject_demos)!=expected:
                raise ValueError('MMLU needs five dev demonstrations per subject or explicit shot count')
            for d in subject_demos:
                messages.extend([dict(role='user',content=choice_prompt(d['question'],d['choices'])),
                                 dict(role='assistant',content=chr(65+int(d['answer'])))])
            prompt=choice_prompt(row['question'],row['choices'])
            target=chr(65+int(row['answer']))
        elif task=='mbpp':
            expected=3 if shots is None else shots
            if len(demos)!=expected or row['task_id'] in {d['task_id'] for d in demos}:
                raise ValueError('MBPP demonstrations missing or overlap evaluation')
            for d in demos:
                messages.extend([dict(role='user',content=mbpp_prompt(d)),dict(role='assistant',content=d['code'])])
            prompt=mbpp_prompt(row); target=None
        elif task=='ifeval':
            # Use the dataset's IFEval prompt unchanged.
            prompt=row['prompt']; target=None
        elif task=='math':
            math_utils=official_metric_module('minerva_math')
            if shots not in (None, 0):
                raise ValueError('MATH nonzero shots require an explicitly registered demonstration set')
            prompt=math_utils.doc_to_text(row)+'\nShow your reasoning and put the final answer in \\boxed{...}.'
            target=math_utils.normalize_final_answer(
                math_utils.remove_boxed(math_utils.last_boxed_only_string(row['solution'])))
        else:
            raise ValueError('unimplemented task '+task)
        messages.append(dict(role='user',content=prompt))
        records.append(dict(doc_id=index, task=task, messages=messages,
            messages_sha256=content_sha(messages), reference=target, record=row,
            id=f"mbpp-{row['task_id']}" if task=='mbpp' else f'{task}-{index}'))
    return records


def trim_at_eos(ids, eos_ids):
    eos={int(x) for x in eos_ids if x is not None}
    return ids[:next((i for i,v in enumerate(ids) if int(v) in eos),len(ids))]


def matched_choice(text):
    """Extract an explicit final option label or one unambiguous leading label."""
    text=text.strip()
    # Parse explicit final-answer labels first.
    explicit=re.findall(r'(?:^|\n)\s*(?:Final\s+answer|Answer|The\s+(?:correct\s+)?answer\s+is)\s*[:：]?\s*\(?([A-E])\)?(?=\s|[.：:]|$)',text,re.I)
    if explicit:
        labels={s.upper() for s in explicit}
        return next(iter(labels)) if len(labels)==1 else None
    # Accept a leading label when the response contains one distinct label.
    labels=re.findall(r'(?:^|\n)\s*\(?([A-E])\)?(?:[.:]\s+|\s*$)',text)
    if len(set(labels))>1:return None
    leading=re.match(r'^\(?([A-E])\)?(?:[.:](?:\s|$)|\s*$)',text)
    return leading.group(1) if leading else None


def score_record(task, record, text, *, choice_parser=None):
    if task=='gsm8k':
        prediction=postprocess(text)
        return dict(prediction=prediction,score=float(equal(prediction,record['reference'])))
    if task in ('arc_challenge','mmlu'):
        if choice_parser=='explicit_final_or_unambiguous_leading_label_v1':
            prediction=matched_choice(text)
            return dict(prediction=prediction,score=float(prediction==record['reference']),invalid=prediction is None)
        if choice_parser is not None:raise ValueError('unknown choice parser')
        matches=re.findall(r'(?:^|\n)\s*(?:Answer\s*:\s*)?\(?([A-E])\)?[.\s]*$',text.strip(),re.I)
        prediction=matches[-1].upper() if matches else None
        return dict(prediction=prediction,score=float(prediction==record['reference']),invalid=prediction is None)
    if task=='mbpp':
        # Return deferred MBPP scoring fields with the generated code.
        return dict(score=None,scored_generation=text,scoring_status='unscored_deferred',
                    sandbox_status='unscored_deferred')
    if task=='ifeval' and not official_scorer_installed():
        # Defer IFEval scoring when its runtime dependencies are unavailable.
        return dict(score=None,scored_generation=text,scoring_status='unscored_deferred',
                    deferred_because='official IFEval scorer is not installed in this runtime')
    if task=='ifeval':
        from langdetect import DetectorFactory
        DetectorFactory.seed=0
        metrics=official_metric_module('ifeval').process_results(record['record'],[text])
        return dict(score=float(metrics['prompt_level_strict_acc']),**metrics)
    if task=='math':
        metrics=official_metric_module('minerva_math').process_results({'answer': record['reference']},[text])
        return dict(score=float(metrics['math_verify']),**metrics)
    raise ValueError('unimplemented scorer')


def scorer_receipt():
    """Return IFEval source hashes, dependency versions, and detector settings."""
    utils=official_metric_module('ifeval')
    folder=Path(utils.__file__).parent
    return dict(lm_eval_version=importlib.metadata.version('lm_eval'),
        dependencies={n:importlib.metadata.version(n) for n in ('langdetect','immutabledict','nltk')},
        language_detector_seed=0,
        files={p.name:file_sha(p) for p in sorted(folder.glob('*.py'))},
        upstream='https://github.com/google-research/google-research/tree/master/instruction_following_eval')
