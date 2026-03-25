interface PageHeaderProps {
  title: string;
  description?: string;
}

export default function PageHeader({ title, description }: PageHeaderProps) {
  return (
    <header className="page-header animate-in">
      <div className="page-header__accent" aria-hidden />
      <div className="page-header__text">
        <h1 className="page-title">{title}</h1>
        {description && <p className="page-desc">{description}</p>}
      </div>
    </header>
  );
}
