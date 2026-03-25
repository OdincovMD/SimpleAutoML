interface SpinnerProps {
  size?: number;
  className?: string;
}

export default function Spinner({ size = 24, className = "" }: SpinnerProps) {
  return (
    <div
      className={`spinner ${className}`}
      style={{ width: size, height: size }}
      role="status"
      aria-label="Загрузка"
    />
  );
}
