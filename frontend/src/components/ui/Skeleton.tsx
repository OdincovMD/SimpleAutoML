export default function Skeleton({
  className = "",
  style,
}: {
  className?: string;
  style?: React.CSSProperties;
}) {
  return (
    <span
      className={`skeleton ${className}`.trim()}
      style={style}
      aria-hidden
    />
  );
}
