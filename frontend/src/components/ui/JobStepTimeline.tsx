import {
  getStepTimelineState,
  labelForStep,
  stepOrderForKind,
  type StepTimelineState,
} from "../../jobSteps";

function stateLabel(state: StepTimelineState): string {
  switch (state) {
    case "done":
      return "Выполнено";
    case "current":
      return "Сейчас";
    case "upcoming":
      return "Далее";
    case "skipped":
      return "Не использовалось";
    default:
      return "";
  }
}

export default function JobStepTimeline({
  kind,
  currentStep,
  history,
  jobStatus,
}: {
  kind: "train" | "inference" | undefined;
  currentStep: string | undefined;
  history: string[] | undefined;
  jobStatus: string;
}) {
  const order = stepOrderForKind(kind);

  return (
    <div className="job-timeline" role="list" aria-label="Этапы задачи">
      {order.map((stepId, index) => {
        const state = getStepTimelineState(
          stepId,
          currentStep,
          history,
          jobStatus
        );
        const ux = labelForStep(stepId);
        const delayMs = Math.min(index * 35, 400);
        return (
          <div
            key={stepId}
            className={`job-timeline__item job-timeline__item--${state}`}
            role="listitem"
            style={{ animationDelay: `${delayMs}ms` }}
          >
            <div className="job-timeline__marker">
              <span className={`job-timeline__node job-timeline__node--${state}`}>
                {state === "done" && "✓"}
                {state === "current" && <span className="job-timeline__pulse" />}
                {state === "upcoming" && " "}
                {state === "skipped" && "—"}
              </span>
            </div>
            <div className="job-timeline__body">
              <div className="job-timeline__head">
                <span className="job-timeline__title">{ux.title}</span>
                <span className={`job-timeline__badge job-timeline__badge--${state}`}>
                  {stateLabel(state)}
                </span>
              </div>
              <p className="job-timeline__hint">{ux.hint}</p>
            </div>
          </div>
        );
      })}
    </div>
  );
}
