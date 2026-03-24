import { useState, useEffect } from "react";
import { useSearchParams } from "react-router-dom";
import { getModelDownloadUrl } from "../api";
import Card from "../components/ui/Card";
import Button from "../components/ui/Button";
import PageHeader from "../components/ui/PageHeader";
import Input from "../components/ui/Input";

const LS_FOLDER = "automl_last_folder_id";

export default function Models() {
  const [searchParams] = useSearchParams();
  const [folderId, setFolderId] = useState(() => {
    const q = searchParams.get("folder");
    if (q) return q;
    try {
      return localStorage.getItem(LS_FOLDER) || "";
    } catch {
      return "";
    }
  });

  useEffect(() => {
    const f = searchParams.get("folder");
    if (f) setFolderId(f);
  }, [searchParams]);
  const [url, setUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setUrl(null);
    setLoading(true);
    try {
      const downloadUrl = await getModelDownloadUrl(folderId.trim());
      setUrl(downloadUrl);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  const copyUrl = async () => {
    if (!url) return;
    try {
      await navigator.clipboard.writeText(url);
    } catch {
      /* ignore */
    }
  };

  return (
    <div className="animate-in">
      <PageHeader
        title="Скачать модель"
        description="Введите folder_id — он возвращается после загрузки датасета."
      />
      <Card>
        <form onSubmit={handleSubmit} style={{ marginBottom: "var(--space-6)" }}>
          <Input
            label="Folder ID"
            value={folderId}
            onChange={(e) => setFolderId(e.target.value)}
            placeholder="folder_id"
            disabled={loading}
            hint="Используйте folder_id из ответа после загрузки"
          />
          <Button type="submit" disabled={loading} style={{ marginTop: "var(--space-4)" }}>
            {loading ? "Загрузка..." : "Получить ссылку"}
          </Button>
        </form>
        {error && (
          <p style={{ margin: 0, color: "var(--error)", fontSize: "14px" }}>
            {error}
          </p>
        )}
        {url && (
          <div
            style={{
              padding: "var(--space-4)",
              background: "var(--accent-subtle)",
              borderRadius: "var(--radius-md)",
              border: "1px solid var(--accent-muted)",
            }}
          >
            <p style={{ margin: 0, fontWeight: 600, fontSize: "14px", marginBottom: "var(--space-2)" }}>
              Ссылка на скачивание готова
            </p>
            <div style={{ display: "flex", gap: "var(--space-2)", alignItems: "center" }}>
              <a
                href={url}
                target="_blank"
                rel="noreferrer"
                className="btn btn--primary"
                style={{ textDecoration: "none" }}
              >
                Скачать модель
              </a>
              <Button variant="secondary" onClick={copyUrl}>
                Копировать ссылку
              </Button>
            </div>
          </div>
        )}
      </Card>
    </div>
  );
}
