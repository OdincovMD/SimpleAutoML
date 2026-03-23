import { useState, useCallback } from "react";
import { Link } from "react-router-dom";
import {
  uploadDataset,
  startJobFromDrive,
  listDriveFolders,
  type DriveFolder,
} from "../api";
import Button from "../components/ui/Button";
import Card from "../components/ui/Card";
import PageHeader from "../components/ui/PageHeader";
import Spinner from "../components/ui/Spinner";
import CodeBlock from "../components/ui/CodeBlock";

type Source = "zip" | "drive";

export default function Upload() {
  const [source, setSource] = useState<Source>("zip");
  const [file, setFile] = useState<File | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [parentName, setParentName] = useState("");
  const [folders, setFolders] = useState<DriveFolder[]>([]);
  const [selectedFolder, setSelectedFolder] = useState<DriveFolder | null>(null);
  const [loadingFolders, setLoadingFolders] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<{
    job_id: string;
    folder_id: string;
    task: string;
  } | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchFolders = async () => {
    setLoadingFolders(true);
    setError(null);
    setFolders([]);
    setSelectedFolder(null);
    try {
      const list = parentName.trim()
        ? await listDriveFolders(parentName.trim(), undefined)
        : await listDriveFolders(undefined, undefined);
      setFolders(list);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoadingFolders(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setResult(null);
    setLoading(true);
    try {
      if (source === "zip") {
        if (!file) return;
        const data = await uploadDataset(file);
        setResult(data);
      } else {
        if (!selectedFolder) return;
        const data = await startJobFromDrive(selectedFolder.id);
        setResult(data);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
    const f = e.dataTransfer.files?.[0];
    if (f?.name.toLowerCase().endsWith(".zip")) setFile(f);
  }, []);

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(true);
  }, []);

  const onDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
  }, []);

  const canSubmit = source === "zip" ? !!file : !!selectedFolder;

  return (
    <div className="animate-in">
      <PageHeader
        title="Загрузка датасета"
        description="Выберите источник данных и запустите обучение модели. Поддерживаются ZIP-архивы и Google Drive."
      />

      <div className="tabs" style={{ marginBottom: "var(--space-6)" }}>
        <button
          type="button"
          className={`tab ${source === "zip" ? "tab--active" : ""}`}
          onClick={() => {
            setSource("zip");
            setError(null);
            setResult(null);
          }}
        >
          ZIP-архив
        </button>
        <button
          type="button"
          className={`tab ${source === "drive" ? "tab--active" : ""}`}
          onClick={() => {
            setSource("drive");
            setError(null);
            setResult(null);
          }}
        >
          Google Drive
        </button>
      </div>

      <Card>
        <form onSubmit={handleSubmit}>
          {source === "zip" && (
            <div style={{ marginBottom: "var(--space-6)" }}>
              <label
                htmlFor="file"
                style={{
                  display: "block",
                  marginBottom: "var(--space-2)",
                  fontWeight: 500,
                  fontSize: "0.9rem",
                }}
              >
                Выберите или перетащите ZIP-архив
              </label>
              <div
                className={`drop-zone ${dragActive ? "drop-zone--active" : ""}`}
                onDrop={onDrop}
                onDragOver={onDragOver}
                onDragLeave={onDragLeave}
                onClick={() => document.getElementById("file")?.click()}
              >
                <input
                  id="file"
                  type="file"
                  accept=".zip"
                  onChange={(e) => setFile(e.target.files?.[0] || null)}
                  disabled={loading}
                  style={{ display: "none" }}
                />
                {file ? (
                  <p style={{ margin: 0, fontWeight: 500, color: "var(--accent)" }}>
                    ✓ {file.name}
                  </p>
                ) : (
                  <p style={{ margin: 0, color: "var(--text-muted)" }}>
                    Перетащите ZIP сюда или нажмите для выбора
                  </p>
                )}
              </div>
              <p
                style={{
                  margin: "var(--space-2) 0 0",
                  fontSize: "13px",
                  color: "var(--text-muted)",
                }}
              >
                Архив должен содержать папку <code style={{ fontFamily: "var(--font-mono)", fontSize: "12px" }}>dataset</code> (классификация или сегментация).
              </p>
            </div>
          )}

          {source === "drive" && (
            <div style={{ marginBottom: "var(--space-6)" }}>
              <label
                htmlFor="parent"
                style={{
                  display: "block",
                  marginBottom: "var(--space-2)",
                  fontWeight: 500,
                  fontSize: "0.9rem",
                }}
              >
                Название папки (оставьте пустым для корня)
              </label>
              <div style={{ display: "flex", gap: "var(--space-2)", marginBottom: "var(--space-4)" }}>
                <input
                  id="parent"
                  type="text"
                  className="input"
                  value={parentName}
                  onChange={(e) => setParentName(e.target.value)}
                  placeholder="Например: Мои датасеты"
                  disabled={loading}
                  style={{ flex: 1 }}
                />
                <Button
                  type="button"
                  variant="secondary"
                  onClick={fetchFolders}
                  disabled={loadingFolders}
                >
                  {loadingFolders ? <Spinner size={18} /> : "Найти"}
                </Button>
              </div>

              {folders.length > 0 && (
                <>
                  <label style={{ display: "block", marginBottom: "var(--space-2)", fontWeight: 500, fontSize: "0.9rem" }}>
                    Выберите датасет
                  </label>
                  <div
                    style={{
                      display: "flex",
                      flexDirection: "column",
                      gap: "var(--space-2)",
                      marginBottom: "var(--space-4)",
                      maxHeight: 220,
                      overflowY: "auto",
                    }}
                  >
                    {folders.map((f) => (
                      <button
                        key={f.id}
                        type="button"
                        onClick={() => setSelectedFolder(selectedFolder?.id === f.id ? null : f)}
                        style={{
                          padding: "var(--space-3) var(--space-4)",
                          textAlign: "left",
                          border: selectedFolder?.id === f.id ? "2px solid var(--accent)" : "1px solid var(--border)",
                          borderRadius: "var(--radius-md)",
                          background: selectedFolder?.id === f.id ? "var(--accent-subtle)" : "var(--bg-elevated)",
                          cursor: "pointer",
                          fontFamily: "var(--font-sans)",
                          fontSize: "14px",
                          transition: "all var(--transition)",
                        }}
                      >
                        {f.name}
                      </button>
                    ))}
                  </div>
                </>
              )}
            </div>
          )}

          <Button type="submit" fullWidth disabled={!canSubmit || loading}>
            {loading ? (
              <>
                <Spinner size={18} />
                Запуск...
              </>
            ) : (
              "Загрузить и обучить"
            )}
          </Button>
        </form>
      </Card>

      {error && (
        <div
          className="card animate-in"
          style={{
            marginTop: "var(--space-6)",
            background: "var(--error-bg)",
            borderColor: "var(--error)",
          }}
        >
          <p style={{ margin: 0, color: "var(--error)" }}>{error}</p>
        </div>
      )}

      {result && (
        <div
          className="card animate-in"
          style={{
            marginTop: "var(--space-6)",
            background: "var(--success-bg)",
            borderColor: "var(--success)",
          }}
        >
          <p style={{ margin: 0, fontWeight: 600, color: "var(--success)" }}>
            ✓ Задача запущена
          </p>
          <p style={{ margin: "var(--space-2) 0 0", fontSize: "14px", color: "var(--text-muted)" }}>
            Тип: {result.task}
          </p>
          <CodeBlock value={result.job_id} />
          <Link
            to={`/jobs?job=${result.job_id}`}
            className="btn btn--primary btn--md"
            style={{ marginTop: "var(--space-4)", textDecoration: "none" }}
          >
            Отслеживать статус →
          </Link>
        </div>
      )}
    </div>
  );
}
