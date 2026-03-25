import React from "react";

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
  hint?: string;
}

export default function Input({
  label,
  error,
  hint,
  id,
  className = "",
  style,
  ...props
}: InputProps) {
  const inputId = id || `input-${Math.random().toString(36).slice(2, 9)}`;
  return (
    <div className={className} style={style}>
      {label && (
        <label
          htmlFor={inputId}
          style={{
            display: "block",
            marginBottom: "0.5rem",
            fontWeight: 500,
            fontSize: "0.9rem",
          }}
        >
          {label}
        </label>
      )}
      <input
        id={inputId}
        className={`input ${error ? "input--error" : ""}`}
        aria-invalid={!!error}
        aria-describedby={hint ? `${inputId}-hint` : undefined}
        {...props}
      />
      {hint && !error && (
        <p
          id={`${inputId}-hint`}
          style={{
            margin: "0.5rem 0 0",
            fontSize: "13px",
            color: "var(--text-muted)",
          }}
        >
          {hint}
        </p>
      )}
      {error && (
        <p
          style={{
            margin: "0.5rem 0 0",
            fontSize: "13px",
            color: "var(--error)",
          }}
        >
          {error}
        </p>
      )}
    </div>
  );
}
