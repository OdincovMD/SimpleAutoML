import { useState, useEffect } from "react";
import { useSearchParams, useNavigate } from "react-router-dom";
import { getJobStatus } from "../api";
import Card from "../components/ui/Card";
import Button from "../components/ui/Button";
import PageHeader from "../components/ui/PageHeader";
import Input from "../components/ui/Input";
import CodeBlock from "../components/ui/CodeBlock";

const LS_JOB = "automl_last_job_id";

const statusConfig: Record<string, { label: string; badge: string }> = {
  PENDING: { label: "В очереди", badge: "badge--muted" },
  STARTED: { label: "Выполняется", badge: "badge--accent" },
  SUCCESS: { label: "Готово", badge: "badge--success" },
  FAILURE: { label: "Ошибка", badge: "badge--error" },
  REVOKED: { label: "Отменено", badge: "badge--muted" },
};

const TERMINAL_STATUSES = new Set(["SUCCESS", "FAILURE", "REVOKED"]);

export default function Jobs() {
  const [searchParams, setSearchParams] = useSearchParams();
  const navigate = useNavigate();
  const jobId = searchParams.get("job");
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

  useEffect(() => {
    if (jobId) setInputId(jobId);
  }, [jobId]);

  useEffect(() => {
    if (!jobId) return;
    let intervalId: ReturnType<typeof setInterval>;
    const poll = async () => {
      try {
        const data = await getJobStatus(jobId);
        setStatus(data.status);
        if (data.status === "SUCCESS") setError(null);
        if (data.status === "FAILURE" && data.error) setError(data.error);
        if (TERMINAL_STATUSES.has(data.status)) {
          clearInterval(intervalId);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      }
    };
    void poll();
    intervalId = setInterval(poll, 2000);
    return () => clearInterval(intervalId);
  }, [jobId]);

  const handleCheck = (e: React.FormEvent) => {
    e.preventDefault();
    if (inputId.trim()) {
      setSearchParams({ job: inputId.trim() });
    }
  };

  if (!jobId) {
    return (
      <div className="animate-in">
        <PageHeader
          title="Задачи"
          description="Введите Job ID или перейдите по ссылке после загрузки датасета."
        />
        <Card>
          <form onSubmit={handleCheck} style={{ display: "flex", gap: "var(--space-2)" }}>
            <Input
              value={inputId}
              onChange={(e) => setInputId(e.target.value)}
              placeholder="Вставьте job_id"
              style={{ flex: 1 }}
            />
            <Button type="submit">Проверить</Button>
          </form>
        </Card>
      </div>
    );
  }

  const cfg = statusConfig[status] || { label: status, badge: "badge--muted" };

  return (
    <div className="animate-in">
      <PageHeader
        title="Статус задачи"
        description={`Job ID: ${jobId}`}
      />
      <Card>
        <div style={{ display: "flex", alignItems: "center", gap: "var(--space-4)", marginBottom: "var(--space-4)" }}>
          <span className={`badge ${cfg.badge}`}>{cfg.label}</span>
          {(status === "PENDING" || status === "STARTED") && (
            <span style={{ fontSize: "13px", color: "var(--text-muted)" }}>
              Обновляется автоматически
            </span>
          )}
        </div>
        <CodeBlock value={jobId} />
        {error && (
          <div
            style={{
              marginTop: "var(--space-4)",
              padding: "var(--space-3)",
              background: "var(--error-bg)",
              borderRadius: "var(--radius-md)",
              color: "var(--error)",
              fontSize: "14px",
            }}
          >
            {error}
          </div>
        )}
        {status === "SUCCESS" && (
          <Button
            style={{ marginTop: "var(--space-4)" }}
            onClick={() => navigate(`/models?folder=${jobId}`)}
          >
            Скачать модель →
          </Button>
        )}
      </Card>
    </div>
  );
}
