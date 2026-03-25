export function formatDateTimeRu(iso: string | null | undefined): string {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return String(iso);
  return d.toLocaleString("ru-RU", {
    dateStyle: "short",
    timeStyle: "short",
  });
}

export const TRAINED_AT_UNKNOWN =
  "Дата неизвестна (модель создана до обновления)";
