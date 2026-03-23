interface PageHeaderProps {
  title: string;
  description?: string;
}

export default function PageHeader({ title, description }: PageHeaderProps) {
  return (
    <header
      className="animate-in"
      style={{ marginBottom: "var(--space-8)" }}
    >
      <h1 className="page-title">{title}</h1>
      {description && <p className="page-desc">{description}</p>}
    </header>
  );
}
