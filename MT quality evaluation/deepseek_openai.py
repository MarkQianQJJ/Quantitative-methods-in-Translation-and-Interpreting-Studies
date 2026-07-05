from pathlib import Path
from openai import OpenAI

client = OpenAI(
    api_key="",
    base_url="https://api.deepseek.com/v1"
)

input_dir = Path(r"D:\vscode\corpus data\deepseek\raw")
output_dir = Path(r"D:\vscode\corpus data\deepseek\translated")
output_dir.mkdir(exist_ok=True, parents=True)

for fpath in input_dir.glob("*.txt"):
    if fpath.stem.startswith("ZH_"):
        continue
    outpath = output_dir / f"ZH_{fpath.name}"
    print(f"翻译 {fpath.name}...")
    paras = [p.strip() for p in fpath.read_text(encoding="utf-8").split("\n\n") if p.strip()]
    result = ""
    for p in paras:
        r = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "Translate English to natural, fluent Chinese"},
                {"role": "user", "content": p}
            ],
            temperature=1.3
        )
        result += r.choices[0].message.content + "\n\n"
    outpath.write_text(result.strip(), encoding="utf-8")
    print(f"完成 {fpath.name}")
