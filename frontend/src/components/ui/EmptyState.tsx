interface EmptyStateProps {
  title: string;
  description?: string;
  icon?: string;
}

export default function EmptyState({ title, description, icon = "📭" }: EmptyStateProps) {
  return (
    <div
      className="card animate-in"
      style={{
        textAlign: "center",
        padding: "var(--space-12)",
        color: "var(--text-muted)",
      }}
    >
      <span style={{ fontSize: "3rem", display: "block", marginBottom: "var(--space-4)" }}>
        {icon}
      </span>
      <p style={{ margin: 0, fontWeight: 600, color: "var(--text)", fontSize: "1.1rem" }}>
        {title}
      </p>
      {description && (
        <p style={{ margin: "0.5rem 0 0", fontSize: "0.95rem" }}>
          {description}
        </p>
      )}
    </div>
  );
}
