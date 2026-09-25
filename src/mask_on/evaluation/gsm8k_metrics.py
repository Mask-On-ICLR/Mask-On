"""Extract and compare GSM8K numeric answers using OpenCompass-style normalization."""
import re


def reference(answer):
    return answer.split('#### ',1)[1].replace(',','').strip()


def postprocess(text):
    text=text.split('Question:',1)[0]
    numbers=re.findall(r'-?\d+\.\d+|-?\d+',text.replace(',',''))
    return numbers[-1] if numbers else 'NULL'


def equal(prediction, target):
    if prediction==target:
        return True
    try:
        return abs(float(prediction)-int(target))<1e-6
    except (TypeError,ValueError):
        return False
