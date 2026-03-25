import { useState, useEffect, useRef, useCallback } from "react";
import { useSearchParams, useNavigate, Link } from "react-router-dom";
import {
  getInferenceDownloadUrl,
  getJobStatus,
  getDatasetMeta,
  listModels,
  isJobTrainComplete,
  type JobProgress,
  type ModelListItem,
  type DatasetMeta,
} from "../api";
import Card from "../components/ui/Card";
import Button from "../components/ui/Button";
import PageHeader from "../components/ui/PageHeader";
import Input from "../components/ui/Input";
import CodeBlock from "../components/ui/CodeBlock";
import JobStepTimeline from "../components/ui/JobStepTimeline";
import MetaRow from "../components/ui/MetaRow";
import Skeleton from "../components/ui/Skeleton";
import {
  TASK_TYPE_UX,
  labelForStep,
  stepProgressPercent,
} from "../jobSteps";
import { formatDateTimeRu, TRAINED_AT_UNKNOWN } from "../format";
import { labels } from "../uiCopy";

const LS_JOB = "automl_last_job_id";

const statusConfig: Record<string, { label: string; badge: string }> = {
  PENDING: { label: "В очереди", badge: "badge--muted" },
  STARTED: { label: "Выполняется", badge: "badge--accent" },
  PROGRESS: { label: "Выполняется", badge: "badge--accent" },
  SUCCESS: { label: "Готово", badge: "badge--success" },
  FAILURE: { label: "Ошибка", badge: "badge--error" },
  REVOKED: { label: "Отменено", badge: "badge--muted" },
};

const TERMINAL_STATUSES = new Set(["SUCCESS", "FAILURE", "REVOKED"]);

function formatElapsed(totalSec: number): string {
  if (totalSec < 60) return `${totalSec} с`;
  const m = Math.floor(totalSec / 60);
  const s = totalSec % 60;
  if (m < 60) return `${m} мин ${s} с`;
  const h = Math.floor(m / 60);
  const mm = m % 60;
  return `${h} ч ${mm} мин`;
}

function ProgressBar({
  status,
  progress,
}: {
  status: string;
  progress: JobProgress | null;
}) {
  const step = progress?.step;
  const ux = step ? labelForStep(step) : null;
  const pct = stepProgressPercent(step, status, progress?.kind);
  const longRunning = ux?.longRunning ?? false;

  if (!ux && status !== "SUCCESS") return null;

  return (
    <>
      <div
        style={{
          height: 10,
          borderRadius: 999,
          background: "var(--bg-muted)",
          overflow: "hidden",
          border: "1px solid var(--border)",
        }}
      >
        <div
          style={{
            width: `${pct}%`,
            height: "100%",
            borderRadius: 999,
            background:
              longRunning && status !== "SUCCESS"
                ? "linear-gradient(90deg, var(--accent), var(--accent-hover, #047857))"
                : "var(--accent)",
            transition: "width 0.45s ease",
          }}
        />
      </div>
      <p
        style={{
          margin: "var(--space-2) 0 0",
          fontSize: "12px",
          color: "var(--text-muted)",
        }}
      >
        {status === "SUCCESS" ? "100%" : `≈ ${pct}%`}
      </p>
      {ux && longRunning && status !== "SUCCESS" && (
        <p
          style={{
            margin: "var(--space-3) 0 0",
            fontSize: "13px",
            padding: "var(--space-2) var(--space-3)",
            background: "var(--accent-subtle, rgba(5,150,105,0.12))",
            borderRadius: "var(--radius-sm)",
            color: "var(--text)",
          }}
        >
          Обычно от нескольких минут до часа и дольше — зависит от размера датасета и CPU. Окно
          можно закрыть: задача выполняется на сервере.
        </p>
      )}
    </>
  );
}

function TrainingOutcomeSection({
  folderId,
  modelRow,
  datasetMeta,
  missingInCatalog,
  onRefresh,
  refreshing,
}: {
  folderId: string;
  modelRow: ModelListItem | null;
  datasetMeta: DatasetMeta | null;
  missingInCatalog: boolean;
  onRefresh: () => void;
  refreshing: boolean;
}) {
  return (
    <div
      style={{
        marginTop: "var(--space-6)",
        padding: "var(--space-5)",
        borderRadius: "var(--radius-md)",
        border: "1px solid var(--accent-muted)",
        background: "var(--accent-subtle)",
      }}
    >
      <h3
        style={{
          margin: "0 0 var(--space-4)",
          fontFamily: "var(--font-heading)",
          fontSize: "1.05rem",
          fontWeight: 700,
          letterSpacing: "-0.02em",
        }}
      >
        Итог обучения
      </h3>

      {missingInCatalog && (
        <p style={{ margin: "0 0 var(--space-3)", fontSize: "14px", color: "var(--text)" }}>
          Запись в каталоге моделей ещё не появилась — иногда нужна секунда после записи в БД.
        </p>
      )}

      {modelRow && (
        <dl className="meta-list" style={{ marginBottom: "var(--space-4)" }}>
          <MetaRow label={labels.trainedAt}>
            {modelRow.trained_at
              ? formatDateTimeRu(modelRow.trained_at)
              : TRAINED_AT_UNKNOWN}
          </MetaRow>
          <MetaRow label="Версия">{modelRow.version}</MetaRow>
          <MetaRow label={labels.imageSizePx}>{modelRow.imgsz}</MetaRow>
          <MetaRow label="Тип задачи">
            {TASK_TYPE_UX[modelRow.task_type ?? ""] ?? modelRow.task_type ?? "—"}
          </MetaRow>
          <MetaRow label="Классов">{modelRow.classes?.length ?? 0}</MetaRow>
          <MetaRow label="Имена классов" mono>
            {modelRow.classes?.length
              ? modelRow.classes.join(", ")
              : "—"}
          </MetaRow>
        </dl>
      )}

      {datasetMeta && (
        <dl className="meta-list" style={{ marginBottom: modelRow ? 0 : "var(--space-4)" }}>
          <MetaRow label="Файлов в датасете (БД)">{datasetMeta.files_total}</MetaRow>
          {datasetMeta.files_pending_train > 0 && (
            <MetaRow label="Ожидают дообучения">
              {datasetMeta.files_pending_train}
            </MetaRow>
          )}
        </dl>
      )}

      <div style={{ display: "flex", flexWrap: "wrap", gap: "var(--space-2)", marginTop: "var(--space-4)" }}>
        {missingInCatalog && (
          <Button type="button" variant="secondary" onClick={onRefresh} disabled={refreshing}>
            {refreshing ? "Обновление…" : "Обновить данные"}
          </Button>
        )}
        <Link
          to={`/models?folder=${encodeURIComponent(folderId)}`}
          className="btn btn--secondary btn--md"
          style={{ textDecoration: "none" }}
        >
          Открыть в каталоге
        </Link>
      </div>
    </div>
  );
}

export default function Jobs() {
  const [searchParams, setSearchParams] = useSearchParams();
  const navigate = useNavigate();
  const jobId = searchParams.get("job");
  const folderFromUrl = searchParams.get("folder");
  const [inputId, setInputId] = useState(() => {
    const fromUrl = searchParams.get("job");
    if (fromUrl) return fromUrl;
    try {
      return localStorage.getItem(LS_JOB) || "";
    } catch {
      return "";
    }
  });
  const [status, setStatus] = useState<string>("");
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<JobProgress | null>(null);
  const [pollReceived, setPollReceived] = useState(false);
  const [elapsedSec, setElapsedSec] = useState(0);
  const startedAtRef = useRef<number | null>(null);
  const [outcomeModel, setOutcomeModel] = useState<ModelListItem | null>(null);
  const [outcomeMeta, setOutcomeMeta] = useState<DatasetMeta | null>(null);
  const [outcomeRefreshKey, setOutcomeRefreshKey] = useState(0);
  const [outcomeLoading, setOutcomeLoading] = useState(false);
  const [completedAt, setCompletedAt] = useState<string | null>(null);

  useEffect(() => {
    if (jobId) setInputId(jobId);
  }, [jobId]);

  useEffect(() => {
    if (!jobId) return;
    setPollReceived(false);
    setStatus("");
    setProgress(null);
    setError(null);
    startedAtRef.current = null;
    setElapsedSec(0);
    setOutcomeModel(null);
    setOutcomeMeta(null);
    setOutcomeRefreshKey(0);
    setOutcomeLoading(false);
    setCompletedAt(null);
  }, [jobId]);

  useEffect(() => {
    if (!jobId) return;
    let cancelled = false;
    let intervalId: ReturnType<typeof setInterval>;
    const poll = async () => {
      if (cancelled) return;
      try {
        const data = await getJobStatus(jobId);
        if (cancelled) return;
        setStatus(data.status);
        setProgress(data.progress);
        setCompletedAt(data.completed_at ?? null);
        if (data.status === "SUCCESS") setError(null);
        if (data.status === "FAILURE" && data.error) setError(data.error);
        if (!startedAtRef.current) startedAtRef.current = Date.now();
        setPollReceived(true);
        if (TERMINAL_STATUSES.has(data.status)) {
          clearInterval(intervalId);
        }
      } catch (err) {
        if (cancelled) return;
        setPollReceived(true);
        setError(err instanceof Error ? err.message : String(err));
      }
    };
    void poll();
    intervalId = setInterval(poll, 2000);
    return () => {
      cancelled = true;
      clearInterval(intervalId);
    };
  }, [jobId]);

  useEffect(() => {
    if (!jobId || !startedAtRef.current) return;
    if (TERMINAL_STATUSES.has(status)) {
      setElapsedSec(
        Math.max(0, Math.floor((Date.now() - startedAtRef.current) / 1000))
      );
      return;
    }
    const tick = () => {
      if (startedAtRef.current) {
        setElapsedSec(Math.floor((Date.now() - startedAtRef.current) / 1000));
      }
    };
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, [jobId, status]);

  const handleCheck = (e: React.FormEvent) => {
    e.preventDefault();
    if (inputId.trim()) {
      const folder = searchParams.get("folder");
      const p = new URLSearchParams({ job: inputId.trim() });
      if (folder?.trim()) p.set("folder", folder.trim());
      setSearchParams(p);
    }
  };

  const copyText = useCallback(async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      /* ignore */
    }
  }, []);

  const folderForModel =
    folderFromUrl?.trim() || progress?.folder_id?.trim() || "";
  const inferenceId =
    status === "SUCCESS" ? progress?.inference_id : undefined;

  const trainComplete = isJobTrainComplete(progress, status);

  useEffect(() => {
    if (!trainComplete || !folderForModel) {
      setOutcomeModel(null);
      setOutcomeMeta(null);
      setOutcomeLoading(false);
      return;
    }
    let cancelled = false;
    setOutcomeLoading(true);
    (async () => {
      try {
        const [models, meta] = await Promise.all([
          listModels(),
          getDatasetMeta(folderForModel).catch(() => null),
        ]);
        if (cancelled) return;
        const row = models.find((m) => m.train_folder === folderForModel) ?? null;
        setOutcomeModel(row);
        setOutcomeMeta(meta);
      } catch {
        if (!cancelled) {
          setOutcomeModel(null);
          setOutcomeMeta(null);
        }
      } finally {
        if (!cancelled) setOutcomeLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [trainComplete, folderForModel, outcomeRefreshKey]);

  const refreshOutcome = () => {
    setOutcomeRefreshKey((k) => k + 1);
  };

  if (!jobId) {
    return (
      <div className="animate-in">
        <PageHeader title={labels.jobsPageTitle} description={labels.jobsPageDesc} />
        <Card>
          <form onSubmit={handleCheck} style={{ display: "flex", gap: "var(--space-2)" }}>
            <Input
              value={inputId}
              onChange={(e) => setInputId(e.target.value)}
              placeholder={labels.jobPlaceholder}
              style={{ flex: 1 }}
            />
            <Button type="submit">Проверить</Button>
          </form>
        </Card>
      </div>
    );
  }

  const cfg = statusConfig[status] || { label: status, badge: "badge--muted" };
  const kind = progress?.kind;
  const showTimeline =
    pollReceived &&
    !(status === "PENDING" && !progress?.step) &&
    (progress != null || status === "SUCCESS");
  const taskLabel = progress?.task_type
    ? TASK_TYPE_UX[progress.task_type] ?? progress.task_type
    : null;

  return (
    <div className="animate-in">
      <PageHeader
        title={labels.taskStatusTitle}
        description={`${labels.taskStatusDescPrefix} ${jobId}`}
      />
      <Card>
        <div
          style={{
            display: "flex",
            flexWrap: "wrap",
            alignItems: "center",
            gap: "var(--space-3)",
            marginBottom: "var(--space-4)",
          }}
        >
          <span className={`badge ${cfg.badge}`}>{cfg.label}</span>
          {startedAtRef.current && (
            <span style={{ fontSize: "13px", color: "var(--text-muted)" }}>
              Прошло: {formatElapsed(elapsedSec)}
            </span>
          )}
          {(status === "PENDING" || status === "STARTED" || status === "PROGRESS") && (
            <span style={{ fontSize: "13px", color: "var(--text-muted)" }}>
              Обновляется автоматически
            </span>
          )}
          {TERMINAL_STATUSES.has(status) && (
            <span style={{ fontSize: "13px", color: "var(--text-muted)" }}>
              {labels.completedAt}:{" "}
              {completedAt
                ? formatDateTimeRu(completedAt)
                : labels.completedAtUnavailable}
            </span>
          )}
        </div>

        {!pollReceived && (
          <div style={{ marginBottom: "var(--space-5)" }}>
            <p style={{ margin: "0 0 var(--space-3)", fontSize: "14px", color: "var(--text-muted)" }}>
              Загрузка статуса…
            </p>
            <Skeleton style={{ width: "100%", height: 12, marginBottom: 8 }} />
            <Skeleton style={{ width: "72%", height: 12 }} />
          </div>
        )}

        {pollReceived && progress && (
          <dl className="meta-list" style={{ marginBottom: "var(--space-5)" }}>
            <MetaRow label="Тип операции">
              {kind === "inference"
                ? "Инференс"
                : kind === "train"
                  ? "Обучение модели"
                  : "—"}
            </MetaRow>
            {taskLabel && <MetaRow label="Задача ML">{taskLabel}</MetaRow>}
            {progress.folder_id && (
              <MetaRow label={labels.projectId} mono>
                <span style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "var(--space-2)" }}>
                  {progress.folder_id}
                  <button
                    type="button"
                    className="btn btn--secondary btn--sm"
                    onClick={() => copyText(progress.folder_id!)}
                  >
                    Копировать
                  </button>
                </span>
              </MetaRow>
            )}
          </dl>
        )}
        {pollReceived && progress?.folder_id && (
          <p style={{ margin: "0 0 var(--space-4)", fontSize: "12px", color: "var(--text-muted)", maxWidth: "52ch" }}>
            {labels.projectIdHint}
          </p>
        )}

        <p style={{ margin: "0 0 var(--space-2)", fontSize: "12px", color: "var(--text-muted)" }}>
          {labels.jobNumber}
        </p>
        <CodeBlock value={jobId} />
        <p style={{ margin: "var(--space-2) 0 0", fontSize: "12px", color: "var(--text-muted)", maxWidth: "52ch" }}>
          {labels.jobNumberHint}
        </p>

        {status === "PENDING" && !progress?.step && (
          <div
            style={{
              marginTop: "var(--space-4)",
              padding: "var(--space-4)",
              background: "var(--bg-muted)",
              borderRadius: "var(--radius-md)",
              border: "1px solid var(--border)",
            }}
          >
            <p style={{ margin: 0, fontSize: "15px", fontWeight: 600 }}>{labels.queueTitle}</p>
            <p style={{ margin: "var(--space-2) 0 0", fontSize: "14px", color: "var(--text-muted)" }}>
              {labels.queueBody}
            </p>
          </div>
        )}

        {showTimeline && (
          <>
            <div style={{ marginTop: "var(--space-5)" }}>
              <ProgressBar status={status} progress={progress} />
            </div>
            {progress?.step && status !== "SUCCESS" && (
              <div
                style={{
                  marginTop: "var(--space-4)",
                  padding: "var(--space-5)",
                  background: "var(--bg-elevated)",
                  borderRadius: "var(--radius-md)",
                  border: "1px solid var(--border)",
                  boxShadow: "var(--shadow-sm, 0 1px 3px rgba(0,0,0,0.06))",
                }}
              >
                <p style={{ margin: 0, fontSize: "18px", fontWeight: 700, letterSpacing: "-0.02em" }}>
                  {labelForStep(progress.step).title}
                </p>
                <p
                  style={{
                    margin: "var(--space-3) 0 0",
                    fontSize: "14px",
                    lineHeight: 1.55,
                    color: "var(--text-muted)",
                  }}
                >
                  {labelForStep(progress.step).hint}
                </p>
              </div>
            )}
            <JobStepTimeline
              kind={kind}
              currentStep={progress?.step}
              history={progress?.steps_history}
              jobStatus={status}
            />
          </>
        )}

        {error && (
          <div
            style={{
              marginTop: "var(--space-5)",
              padding: "var(--space-4)",
              background: "var(--error-bg)",
              borderRadius: "var(--radius-md)",
              border: "1px solid var(--error)",
            }}
          >
            <p style={{ margin: "0 0 var(--space-2)", fontWeight: 600, color: "var(--error)" }}>
              Не удалось выполнить задачу
            </p>
            <pre
              style={{
                margin: 0,
                padding: "var(--space-3)",
                background: "var(--bg-elevated)",
                borderRadius: "var(--radius-sm)",
                fontSize: "13px",
                fontFamily: "var(--font-mono)",
                whiteSpace: "pre-wrap",
                wordBreak: "break-word",
                color: "var(--text)",
              }}
            >
              {error}
            </pre>
            <div style={{ display: "flex", flexWrap: "wrap", gap: "var(--space-2)", marginTop: "var(--space-3)" }}>
              <Button type="button" variant="secondary" onClick={() => copyText(error)}>
                Копировать текст ошибки
              </Button>
              <Link to="/upload" className="btn btn--secondary btn--md" style={{ textDecoration: "none" }}>
                Новая загрузка
              </Link>
              <Link to="/models" className="btn btn--secondary btn--md" style={{ textDecoration: "none" }}>
                Модели
              </Link>
            </div>
          </div>
        )}

        {trainComplete && folderForModel && (
          <TrainingOutcomeSection
            folderId={folderForModel}
            modelRow={outcomeModel}
            datasetMeta={outcomeMeta}
            missingInCatalog={!outcomeLoading && outcomeModel === null}
            onRefresh={refreshOutcome}
            refreshing={outcomeLoading}
          />
        )}

        {status === "SUCCESS" && inferenceId && folderForModel && (
          <Button
            style={{ marginTop: "var(--space-4)" }}
            onClick={async () => {
              try {
                const url = await getInferenceDownloadUrl(folderForModel, inferenceId);
                window.open(url, "_blank", "noopener,noreferrer");
              } catch (err) {
                setError(err instanceof Error ? err.message : String(err));
              }
            }}
          >
            Скачать архив результатов →
          </Button>
        )}
        {status === "SUCCESS" && !inferenceId && folderForModel && (
          <Button
            style={{ marginTop: "var(--space-4)" }}
            onClick={() =>
              navigate(`/models?folder=${encodeURIComponent(folderForModel)}`)
            }
          >
            Скачать модель →
          </Button>
        )}
        {status === "SUCCESS" && !folderForModel && (
          <p style={{ marginTop: "var(--space-4)", fontSize: "13px", color: "var(--text-muted)" }}>
            Укажите ID проекта в адресе страницы (<code style={{ fontFamily: "var(--font-mono)" }}>?folder=…</code>)
            или выберите модель на странице «Модели».
          </p>
        )}
      </Card>
    </div>
  );
}
