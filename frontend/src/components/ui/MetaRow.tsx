export default function MetaRow({
  label,
  children,
  mono,
}: {
  label: string;
  children: React.ReactNode;
  mono?: boolean;
}) {
  return (
    <>
      <dt className="meta-row__label">{label}</dt>
      <dd
        className={`meta-row__value${mono ? " meta-row__value--mono" : ""}`}
      >
        {children}
      </dd>
    </>
  );
}
