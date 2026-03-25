import { useState, useEffect } from "react";
import { Link, useSearchParams, useNavigate } from "react-router-dom";
import {
  getModelDownloadUrl,
  listModels,
  retrainDataset,
  getDatasetMeta,
  type ModelListItem,
} from "../api";
import Card from "../components/ui/Card";
import Button from "../components/ui/Button";
import PageHeader from "../components/ui/PageHeader";
import Input from "../components/ui/Input";
import Skeleton from "../components/ui/Skeleton";
import { TASK_TYPE_UX } from "../jobSteps";
import { formatDateTimeRu, TRAINED_AT_UNKNOWN } from "../format";
import { labels } from "../uiCopy";

const LS_FOLDER = "automl_last_folder_id";

function taskLabel(t: string | null) {
  if (!t) return "—";
  return TASK_TYPE_UX[t] ?? t;
}

export default function Models() {
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
  const [catalog, setCatalog] = useState<ModelListItem[]>([]);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [loadingCatalog, setLoadingCatalog] = useState(true);

  const [url, setUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const [retrainFile, setRetrainFile] = useState<File | null>(null);
  const [retrainFolder, setRetrainFolder] = useState("");
  const [retrainTaskTypePick, setRetrainTaskTypePick] = useState("");
  const [retrainBusy, setRetrainBusy] = useState(false);
  const [metaByFolder, setMetaByFolder] = useState<
    Record<string, { pending: number; total: number }>
  >({});

  useEffect(() => {
    const f = searchParams.get("folder");
    if (f) setFolderId(f);
  }, [searchParams]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      setLoadingCatalog(true);
      setCatalogError(null);
      try {
        const rows = await listModels();
        if (!cancelled) setCatalog(rows);
      } catch (err) {
        if (!cancelled) {
          setCatalogError(
            err instanceof Error ? err.message : String(err)
          );
        }
      } finally {
        if (!cancelled) setLoadingCatalog(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!catalog.length) return;
    let cancelled = false;
    (async () => {
      const next: Record<string, { pending: number; total: number }> = {};
      await Promise.all(
        catalog.map(async (m) => {
          try {
            const meta = await getDatasetMeta(m.train_folder);
            next[m.train_folder] = {
              pending: meta.files_pending_train,
              total: meta.files_total,
            };
          } catch {
            next[m.train_folder] = { pending: 0, total: 0 };
          }
        })
      );
      if (!cancelled) setMetaByFolder(next);
    })();
    return () => {
      cancelled = true;
    };
  }, [catalog]);

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

  const submitRetrain = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!retrainFile || !retrainFolder.trim()) return;
    const row = catalog.find((x) => x.train_folder === retrainFolder.trim());
    const taskType =
      row?.task_type?.trim() || retrainTaskTypePick.trim();
    if (!taskType) {
      setError(
        "Укажите тип задачи (сегментация или классификация), если он не сохранён в каталоге."
      );
      return;
    }
    setRetrainBusy(true);
    setError(null);
    try {
      const res = await retrainDataset(retrainFolder.trim(), retrainFile, taskType);
      try {
        localStorage.setItem(LS_FOLDER, res.folder_id);
      } catch {
        /* ignore */
      }
      navigate(
        `/jobs?job=${encodeURIComponent(res.job_id)}&folder=${encodeURIComponent(res.folder_id)}`
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setRetrainBusy(false);
      setRetrainFile(null);
    }
  };

  return (
    <div className="animate-in">
      <PageHeader title="Модели" description={labels.modelsPageDesc} />

      <Card style={{ marginBottom: "var(--space-8)" }}>
        <h2
          style={{
            margin: "0 0 var(--space-4)",
            fontSize: "1.05rem",
            fontWeight: 700,
          }}
        >
          Каталог
        </h2>
        {loadingCatalog && (
          <div className="model-catalog-skeleton" aria-busy="true" aria-label="Загрузка каталога">
            {[0, 1, 2].map((i) => (
              <div key={i} className="model-catalog-skeleton__row">
                <Skeleton style={{ width: "58%", height: 14 }} />
                <div style={{ display: "flex", gap: "var(--space-2)", flexWrap: "wrap" }}>
                  <Skeleton style={{ width: 56, height: 24, borderRadius: 999 }} />
                  <Skeleton style={{ width: 120, height: 24, borderRadius: 999 }} />
                  <Skeleton style={{ width: 88, height: 24, borderRadius: 999 }} />
                </div>
                <Skeleton style={{ width: "100%", height: 12 }} />
              </div>
            ))}
          </div>
        )}
        {catalogError && (
          <p style={{ margin: 0, color: "var(--error)", fontSize: "14px" }}>
            {catalogError}
          </p>
        )}
        {!loadingCatalog && !catalogError && catalog.length === 0 && (
          <p style={{ margin: 0, color: "var(--text-muted)", fontSize: "14px" }}>
            Пока нет записей в БД. Загрузите датасет и дождитесь завершения обучения.
          </p>
        )}
        {!loadingCatalog && catalog.length > 0 && (
          <ul
            style={{
              listStyle: "none",
              margin: 0,
              padding: 0,
              display: "flex",
              flexDirection: "column",
              gap: "var(--space-4)",
            }}
          >
            {catalog.map((m) => {
              const meta = metaByFolder[m.train_folder];
              return (
                <li
                  key={m.train_folder}
                  style={{
                    padding: "var(--space-5)",
                    borderRadius: "var(--radius-md)",
                    border: "1px solid var(--border)",
                    background: "var(--bg-elevated)",
                    boxShadow: "var(--shadow-sm)",
                  }}
                >
                  <code
                    style={{
                      display: "block",
                      fontFamily: "var(--font-mono)",
                      fontSize: "13px",
                      wordBreak: "break-all",
                      marginBottom: "var(--space-3)",
                      color: "var(--text)",
                    }}
                  >
                    {m.train_folder}
                  </code>
                  <div
                    style={{
                      display: "flex",
                      flexWrap: "wrap",
                      gap: "var(--space-2)",
                      alignItems: "center",
                    }}
                  >
                    <span className="badge badge--muted">v{m.version}</span>
                    <span className="badge badge--accent">{taskLabel(m.task_type)}</span>
                    <span
                      className="badge badge--muted"
                      title={labels.imageSizeHint}
                    >
                      {m.imgsz} px
                    </span>
                  </div>
                  <p
                    style={{
                      margin: "var(--space-2) 0 0",
                      fontSize: "13px",
                      color: "var(--text-muted)",
                    }}
                  >
                    {labels.trainedAt}:{" "}
                    {m.trained_at ? formatDateTimeRu(m.trained_at) : TRAINED_AT_UNKNOWN}
                  </p>
                  {meta !== undefined && (
                    <p
                      style={{
                        margin: "var(--space-3) 0 0",
                        fontSize: "13px",
                        color: "var(--text-muted)",
                      }}
                    >
                      Файлов в БД: {meta.total}
                      {meta.pending > 0
                        ? ` · ожидают дообучения: ${meta.pending}`
                        : ""}
                    </p>
                  )}
                  <details
                    style={{
                      marginTop: "var(--space-3)",
                      borderTop: "1px solid var(--border)",
                      paddingTop: "var(--space-3)",
                    }}
                  >
                    <summary
                      className="model-details-summary"
                      style={{
                        cursor: "pointer",
                        fontWeight: 600,
                        fontSize: "13px",
                        color: "var(--text)",
                        listStyle: "none",
                      }}
                    >
                      Классы ({m.classes?.length ?? 0})
                    </summary>
                    <p
                      style={{
                        margin: "var(--space-2) 0 0",
                        fontSize: "13px",
                        lineHeight: 1.65,
                        color: "var(--text-muted)",
                      }}
                    >
                      {m.classes?.length ? m.classes.join(", ") : "—"}
                    </p>
                  </details>
                  <div
                    style={{
                      display: "flex",
                      flexWrap: "wrap",
                      gap: "var(--space-2)",
                      marginTop: "var(--space-4)",
                    }}
                  >
                    <Button
                      variant="secondary"
                      type="button"
                      onClick={() => {
                        setFolderId(m.train_folder);
                        setUrl(null);
                        setError(null);
                      }}
                    >
                      Ссылка на веса
                    </Button>
                    <Link
                      to={`/inference?folder=${encodeURIComponent(m.train_folder)}`}
                      className="btn btn--secondary btn--md"
                      style={{ textDecoration: "none" }}
                    >
                      Инференс
                    </Link>
                    <Button
                      variant="secondary"
                      type="button"
                      onClick={() => {
                        setRetrainFolder(m.train_folder);
                        setRetrainTaskTypePick("");
                        setError(null);
                      }}
                    >
                      Дообучить…
                    </Button>
                  </div>
                </li>
              );
            })}
          </ul>
        )}
      </Card>

      {retrainFolder && (
        <Card style={{ marginBottom: "var(--space-8)" }}>
          <h2
            style={{
              margin: "0 0 var(--space-3)",
              fontSize: "1.05rem",
              fontWeight: 700,
            }}
          >
            Дообучение: {retrainFolder}
          </h2>
          <p
            style={{
              margin: "0 0 var(--space-4)",
              fontSize: "14px",
              color: "var(--text-muted)",
            }}
          >
            ZIP с новыми данными (структура с папкой{" "}
            <code style={{ fontFamily: "var(--font-mono)", fontSize: "12px" }}>dataset</code>
            ) будет слит с существующим проектом.
          </p>
          <form onSubmit={submitRetrain}>
            {(() => {
              const row = catalog.find((x) => x.train_folder === retrainFolder);
              const needsPick = !row?.task_type?.trim();
              if (!needsPick) return null;
              return (
                <div style={{ marginBottom: "var(--space-4)" }}>
                  <label
                    style={{
                      display: "block",
                      fontSize: "13px",
                      fontWeight: 600,
                      marginBottom: "var(--space-2)",
                    }}
                  >
                    Тип задачи
                  </label>
                  <select
                    value={retrainTaskTypePick}
                    onChange={(e) => setRetrainTaskTypePick(e.target.value)}
                    disabled={retrainBusy}
                    required
                    style={{
                      width: "100%",
                      maxWidth: 320,
                      padding: "var(--space-2) var(--space-3)",
                      borderRadius: "var(--radius-md)",
                      border: "1px solid var(--border)",
                      background: "var(--bg-elevated)",
                      fontSize: "14px",
                    }}
                  >
                    <option value="">Выберите…</option>
                    <option value="классификация">
                      {TASK_TYPE_UX["классификация"] ?? "классификация"}
                    </option>
                    <option value="сегментация">
                      {TASK_TYPE_UX["сегментация"] ?? "сегментация"}
                    </option>
                  </select>
                  <p
                    style={{
                      margin: "var(--space-2) 0 0",
                      fontSize: "12px",
                      color: "var(--text-muted)",
                    }}
                  >
                    В каталоге нет сохранённого типа — укажите вручную для API дообучения.
                  </p>
                </div>
              );
            })()}
            <input
              type="file"
              accept=".zip"
              onChange={(e) => setRetrainFile(e.target.files?.[0] || null)}
              disabled={retrainBusy}
              style={{ marginBottom: "var(--space-3)", fontSize: "14px" }}
            />
            <div style={{ display: "flex", gap: "var(--space-2)", flexWrap: "wrap" }}>
              <Button type="submit" disabled={!retrainFile || retrainBusy}>
                {retrainBusy ? "Запуск…" : "Запустить дообучение"}
              </Button>
              <Button
                type="button"
                variant="secondary"
                onClick={() => setRetrainFolder("")}
              >
                Отмена
              </Button>
            </div>
          </form>
        </Card>
      )}

      <PageHeader
        title={labels.downloadWeightsSection}
        description={labels.downloadWeightsDesc}
      />
      <Card>
        <form onSubmit={handleSubmit} style={{ marginBottom: "var(--space-6)" }}>
          <Input
            label={labels.projectIdField}
            value={folderId}
            onChange={(e) => setFolderId(e.target.value)}
            placeholder={labels.projectIdPlaceholder}
            disabled={loading}
            hint="Тот же идентификатор, что в каталоге выше или после загрузки датасета."
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
