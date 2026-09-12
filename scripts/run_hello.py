"""Run a hello-world end-to-end pass through pipeline stubs (Phase 0 exit criteria)."""
import json
from preprocessing.pipeline import pipeline_stub

def main():
    dummy = 'This is a dummy input: hello world.'
    out = pipeline_stub(dummy)
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
