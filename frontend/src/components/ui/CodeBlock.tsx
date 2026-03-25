import { useState } from "react";

interface CodeBlockProps {
  value: string;
}

export default function CodeBlock({ value }: CodeBlockProps) {
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      /* ignore */
    }
  };

  return (
    <div className="code-block-row">
      <code className="code-block">{value}</code>
      <button
        type="button"
        onClick={copy}
        className="btn btn--secondary btn--sm"
        title="Копировать"
      >
        {copied ? "Скопировано" : "Копия"}
      </button>
    </div>
  );
}
