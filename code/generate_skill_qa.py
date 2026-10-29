# Phase 3, Step 3.1: Generate Skill QA Pairs (Racket / Python / Cross-paradigm)

import os
# Fix SSL cert issue
if 'SSL_CERT_FILE' in os.environ:
    del os.environ['SSL_CERT_FILE']

from openai import OpenAI
import json, jsonlines, time, random
from config import DEEPSEEK_API_KEY, MODEL

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

os.makedirs("knot_data/knot_skill", exist_ok=True)

def llm_call(prompt, max_tokens=1500, temperature=0.7):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature
    )
    return resp.choices[0].message.content

def parse_qa_json(text):
    import re
    text = re.sub(r'```(?:json)?\s*', '', text).strip()
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            return []
    return []

ALGO_TASKS = [
    "compute factorial of n", "reverse a list", "check if a number is prime",
    "find the nth Fibonacci number", "compute the GCD of two numbers",
    "flatten a nested list", "check if a string is a palindrome",
    "implement binary search", "compute the sum of a list recursively",
    "find all permutations of a list", "implement quicksort", "implement mergesort",
    "compute power set of a set", "count occurrences of an element in a list",
    "find the depth of a binary tree", "implement a queue using two stacks",
    "compute the longest common subsequence of two strings",
    "check if two strings are anagrams", "find duplicate elements in a list",
    "implement run-length encoding", "compute the Levenshtein distance",
    "find the majority element in a list", "implement a simple tokenizer",
    "compute prefix sums of a list", "find all subsets of a list",
    "implement memoized Fibonacci", "compute Collatz sequence for n",
    "implement insertion sort", "find the kth largest element",
    "group elements by a key function",
]

FORGET_PROMPT = """Generate {n} question-answer pairs where the question asks to implement an algorithm in Racket (a Lisp dialect), and the answer provides a correct Racket implementation.

Algorithm tasks to cover: {tasks}

Requirements:
- Each question must specify the task and that the answer should be in Racket
- The Racket answer must use idiomatic Racket: define, lambda, let/let*, car/cdr/cons, recursion, map/filter/foldl
- Demonstrate Racket-specific features where appropriate (tail recursion with named let, quasiquote, etc.)
- Return ONLY a JSON array: [{{"question": "Implement [task] in Racket", "answer": "(define ...)", "racket_features": ["feature1", "feature2"]}}]"""

RETAIN_PROMPT = """Generate {n} question-answer pairs where the question asks to implement an algorithm in Python, and the answer provides a correct Python implementation.

Algorithm tasks to cover: {tasks}

Requirements:
- Clean, readable Python using standard library only
- Use descriptive variable names and docstrings
- Return ONLY a JSON array: [{{"question": "Implement [task] in Python", "answer": "def ...", "complexity": "O(...)"}}]"""

BOUNDARY_PROMPT = """Generate {n} boundary question-answer pairs about algorithm design that require understanding BOTH functional programming concepts (as used in Racket/Lisp) AND imperative programming concepts (as used in Python).

Examples of good boundary questions:
- "Explain two different ways to implement tail-recursive factorial: one in a functional style (using accumulators as in Racket) and one in an iterative style (as in Python). What is the fundamental trade-off?"
- "How does the concept of 'cons cell' in Racket/Lisp correspond to Python's list implementation? What are the performance implications for prepending vs appending?"

Requirements:
- Questions must require knowledge of BOTH paradigms to answer fully
- Answers should explicitly compare functional and imperative approaches
- Return ONLY a JSON array: [{{"question": "...", "answer": "...", "functional_aspect": "...", "imperative_aspect": "..."}}]"""

all_forget, all_retain, all_boundary = [], [], []

# Forget QA (Racket, target 1500) - reduced for testing
print("Generating Forget QA (Racket)...")
for i in range(20):
    tasks_batch = random.sample(ALGO_TASKS, min(5, len(ALGO_TASKS)))
    try:
        raw = llm_call(FORGET_PROMPT.format(n=5, tasks=", ".join(tasks_batch)), max_tokens=2000)
        qas = parse_qa_json(raw)
        for qa in qas:
            qa["split"] = "forget"
            qa["skill_type"] = "racket"
        all_forget.extend(qas)
        print(f"  forget batch {i}: {len(qas)} items")
    except Exception as ex:
        print(f"  Forget batch {i} error: {ex}")
    time.sleep(0.5)

# Retain QA (Python, target 5000) - reduced for testing
print("Generating Retain QA (Python)...")
for i in range(30):
    tasks_batch = random.sample(ALGO_TASKS, min(5, len(ALGO_TASKS)))
    try:
        raw = llm_call(RETAIN_PROMPT.format(n=5, tasks=", ".join(tasks_batch)), max_tokens=2000)
        qas = parse_qa_json(raw)
        for qa in qas:
            qa["split"] = "retain"
            qa["skill_type"] = "python"
        all_retain.extend(qas)
        print(f"  retain batch {i}: {len(qas)} items")
    except Exception as ex:
        print(f"  Retain batch {i} error: {ex}")
    time.sleep(0.5)

# Boundary QA (cross-paradigm, target 1000) - reduced for testing
print("Generating Boundary QA...")
for i in range(10):
    try:
        raw = llm_call(BOUNDARY_PROMPT.format(n=5), max_tokens=2000)
        qas = parse_qa_json(raw)
        for qa in qas:
            qa["split"] = "boundary"
            qa["skill_type"] = "cross_paradigm"
        all_boundary.extend(qas)
        print(f"  boundary batch {i}: {len(qas)} items")
    except Exception as ex:
        print(f"  Boundary batch {i} error: {ex}")
    time.sleep(0.5)

with jsonlines.open("knot_data/knot_skill/forget_raw.jsonl", "w") as w:
    w.write_all(all_forget)
with jsonlines.open("knot_data/knot_skill/retain_raw.jsonl", "w") as w:
    w.write_all(all_retain)
with jsonlines.open("knot_data/knot_skill/boundary_raw.jsonl", "w") as w:
    w.write_all(all_boundary)

print(f"Total: {len(all_forget)} forget, {len(all_retain)} retain, {len(all_boundary)} boundary")
