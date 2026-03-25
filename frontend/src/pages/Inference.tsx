import { useState, useEffect, useCallback } from "react";
import { Link, useSearchParams, useNavigate } from "react-router-dom";
import { listModels, startInference, type ModelListItem } from "../api";
import Card from "../components/ui/Card";
import Button from "../components/ui/Button";
import PageHeader from "../components/ui/PageHeader";
import Input from "../components/ui/Input";
import { TASK_TYPE_UX } from "../jobSteps";
import { labels } from "../uiCopy";

const LS_FOLDER = "automl_last_folder_id";

export default function Inference() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const [folderId, setFolderId] = useState(() => {
    const q = searchParams.get("folder");
    if (q) return q;
    try {
      return localStorage.getItem(LS_FOLDER) || "";
    } catch {
      return "";
    }
  });
  const [taskType, setTaskType] = useState("классификация");
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [catalog, setCatalog] = useState<ModelListItem[]>([]);

  useEffect(() => {
    const f = searchParams.get("folder");
    if (f) setFolderId(f);
  }, [searchParams]);

  const loadCatalog = useCallback(async () => {
    try {
      const rows = await listModels();
      setCatalog(rows);
    } catch {
      setCatalog([]);
    }
  }, []);

  useEffect(() => {
    void loadCatalog();
  }, [loadCatalog]);

  useEffect(() => {
    if (!catalog.length || folderId.trim()) return;
    const first = catalog[0];
    if (first?.task_type) setTaskType(first.task_type);
  }, [catalog, folderId]);

  const copyProjectId = async (id: string) => {
    try {
      await navigator.clipboard.writeText(id);
    } catch {
      /* ignore */
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    if (!file || !folderId.trim()) return;
    setLoading(true);
    try {
      const res = await startInference(folderId.trim(), taskType, file);
      navigate(
        `/jobs?job=${encodeURIComponent(res.job_id)}&folder=${encodeURIComponent(res.folder_id)}`
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="animate-in">
      <PageHeader title="Инференс" description={labels.inferencePageDesc} />

      {catalog.length > 0 && (
        <Card style={{ marginBottom: "var(--space-6)" }}>
          <p
            style={{
              margin: "0 0 var(--space-2)",
              fontSize: "14px",
              fontWeight: 600,
            }}
          >
            {labels.pickModel}
          </p>
          <p
            style={{
              margin: "0 0 var(--space-4)",
              fontSize: "13px",
              color: "var(--text-muted)",
            }}
          >
            Наведите на кнопку проекта — покажем полный ID. Рядом можно скопировать ID в буфер.
          </p>
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              gap: "var(--space-3)",
            }}
          >
            {catalog.map((m) => (
              <div
                key={m.train_folder}
                style={{
                  display: "flex",
                  flexWrap: "wrap",
                  alignItems: "center",
                  gap: "var(--space-2)",
                }}
              >
                <Button
                  type="button"
                  variant="secondary"
                  title={`Полный ${labels.projectId}:\n${m.train_folder}`}
                  onClick={() => {
                    setFolderId(m.train_folder);
                    if (m.task_type) setTaskType(m.task_type);
                  }}
                >
                  Проект {m.train_folder.slice(0, 8)}…
                </Button>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={() => copyProjectId(m.train_folder)}
                >
                  {labels.copyProjectId}
                </Button>
              </div>
            ))}
          </div>
        </Card>
      )}

      <Card>
        <form onSubmit={handleSubmit}>
          <Input
            label={labels.projectIdField}
            value={folderId}
            onChange={(e) => setFolderId(e.target.value)}
            placeholder={labels.projectIdPlaceholder}
            disabled={loading}
            hint={labels.projectIdHint}
          />
          <div style={{ marginTop: "var(--space-4)" }}>
            <label
              htmlFor="infer-task"
              style={{
                display: "block",
                marginBottom: "var(--space-2)",
                fontWeight: 500,
                fontSize: "0.9rem",
              }}
            >
              Тип задачи
            </label>
            <select
              id="infer-task"
              className="input"
              value={taskType}
              onChange={(e) => setTaskType(e.target.value)}
              disabled={loading}
              style={{ width: "100%", maxWidth: 360 }}
            >
              <option value="классификация">
                {TASK_TYPE_UX["классификация"]}
              </option>
              <option value="сегментация">{TASK_TYPE_UX["сегментация"]}</option>
            </select>
          </div>
          <div style={{ marginTop: "var(--space-4)" }}>
            <label
              htmlFor="infer-zip"
              style={{
                display: "block",
                marginBottom: "var(--space-2)",
                fontWeight: 500,
                fontSize: "0.9rem",
              }}
            >
              ZIP с изображениями
            </label>
            <input
              id="infer-zip"
              type="file"
              accept=".zip"
              disabled={loading}
              onChange={(e) => setFile(e.target.files?.[0] || null)}
            />
          </div>
          {error && (
            <p
              style={{
                margin: "var(--space-4) 0 0",
                color: "var(--error)",
                fontSize: "14px",
              }}
            >
              {error}
            </p>
          )}
          <Button
            type="submit"
            disabled={!file || !folderId.trim() || loading}
            style={{ marginTop: "var(--space-5)" }}
          >
            {loading ? "Отправка…" : "Запустить инференс"}
          </Button>
        </form>
        <p
          style={{
            margin: "var(--space-5) 0 0",
            fontSize: "13px",
            color: "var(--text-muted)",
          }}
        >
          После запуска откроется страница задачи. Когда статус «Готово», нажмите «Скачать архив результатов».
        </p>
        <Link
          to="/jobs"
          style={{
            display: "inline-block",
            marginTop: "var(--space-3)",
            fontSize: "14px",
            color: "var(--accent)",
          }}
        >
          Все задачи →
        </Link>
      </Card>
    </div>
  );
}
