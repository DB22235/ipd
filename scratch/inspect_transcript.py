import json

log_path = r'C:\Users\Dhruv Dube\.gemini\antigravity-ide\brain\2f1d830d-7938-4871-ba6b-3a9d4b8a6b00\.system_generated\logs\transcript.jsonl'
with open(log_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i in range(2917, min(2935, len(lines))):
    d = json.loads(lines[i])
    t = d.get('type')
    s = d.get('source')
    content = str(d.get('content', ''))
    tc = d.get('tool_calls', [])
    print(f"\n--- Line {i} | Source: {s} | Type: {t} ---")
    if tc:
        print(f"Tool calls: {[call.get('name') for call in tc]}")
    if content:
        print(f"Content: {content[:600]}")
