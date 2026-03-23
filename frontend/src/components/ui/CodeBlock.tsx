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
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: "var(--space-2)",
        marginTop: "var(--space-2)",
      }}
    >
      <code
        style={{
          flex: 1,
          fontFamily: "var(--font-mono)",
          fontSize: "13px",
          background: "var(--bg-muted)",
          padding: "8px 12px",
          borderRadius: "var(--radius-sm)",
          overflow: "hidden",
          textOverflow: "ellipsis",
        }}
      >
        {value}
      </code>
      <button
        type="button"
        onClick={copy}
        className="btn btn--secondary btn--sm"
        title="Копировать"
      >
        {copied ? "✓" : "📋"}
      </button>
    </div>
  );
}
