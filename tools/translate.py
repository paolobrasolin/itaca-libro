import argparse
import sys
import openai
import tiktoken


def parse_args():
    parser = argparse.ArgumentParser(
        description="Translate LaTeX files using OpenAI's API."
    )
    parser.add_argument(
        "--source", type=str, help="Input LaTeX file (or stdin if omitted)"
    )
    parser.add_argument(
        "--target", type=str, help="Output LaTeX file (or stdout if omitted)"
    )
    parser.add_argument(
        "--from", dest="source_lang", required=True, help="Source language"
    )
    parser.add_argument(
        "--to", dest="target_lang", required=True, help="Target language"
    )
    return parser.parse_args()


def estimate_cost(text):
    encoder = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoder.encode(text))
    cost_per_1m_tokens = 0.15  # Adjust based on OpenAI's pricing
    estimated_cost = num_tokens * cost_per_1m_tokens / 1000000
    return num_tokens, estimated_cost


def translate_text(text, source_lang, target_lang):
    prompt = f"You are a professional translator. Translate the following LaTeX document from {source_lang} to {target_lang}, preserving all LaTeX commands and formatting. Only translate the human-readable text, keeping LaTeX commands untouched.\n\nOriginal ({source_lang}):\n{text}\n\nTranslated ({target_lang}):"

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": prompt}],
        max_tokens=16384,
    )

    return response.choices[0].message.content.strip()


def main():
    args = parse_args()

    # Read input
    if args.source:
        with open(args.source, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = sys.stdin.read()

    # Estimate cost
    num_tokens, estimated_cost = estimate_cost(text)
    print(f"Estimated token count: {num_tokens}")
    print(f"Estimated cost: ${estimated_cost:.4f}")

    confirm = input("Proceed with translation? (yes/no): ").strip().lower()
    if confirm != "yes":
        print("Translation aborted.")
        sys.exit(0)

    # Translate
    translated_text = translate_text(text, args.source_lang, args.target_lang)

    # Output result
    if args.target:
        with open(args.target, "w", encoding="utf-8") as f:
            f.write(translated_text)
    else:
        print(translated_text)


if __name__ == "__main__":
    main()
