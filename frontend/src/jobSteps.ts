/** @see backend.app.job_progress.STEP_ORDER */
export const JOB_STEP_ORDER = [
  "job_queued",
  "indexing_files",
  "splitting",
  "learning",
  "saving_model",
  "fine_tune_split",
  "fine_tuning",
  "fine_tune_saved",
  "nothing_to_train",
  "saving_to_cloud",
  "job_complete",
] as const;

export type JobStepId = (typeof JOB_STEP_ORDER)[number];

export const JOB_STEP_UX: Record<
  string,
  { title: string; hint: string; longRunning?: boolean }
> = {
  job_queued: {
    title: "Задача в очереди",
    hint: "Worker принял задание, скоро начнётся обработка.",
  },
  indexing_files: {
    title: "Учёт файлов",
    hint: "Сохраняем список файлов датасета в базе данных.",
  },
  splitting: {
    title: "Подготовка выборки",
    hint: "Делим изображения на обучение и проверку (train / val).",
  },
  learning: {
    title: "Обучение модели",
    hint: "Самый долгий этап: нейросеть учится на ваших данных. Окно можно закрыть — процесс идёт на сервере.",
    longRunning: true,
  },
  saving_model: {
    title: "Сохранение весов",
    hint: "Записываем обученную модель на диск.",
  },
  fine_tune_split: {
    title: "Подготовка к дообучению",
    hint: "Обновляем разбиение данных для новой порции изображений.",
  },
  fine_tuning: {
    title: "Дообучение",
    hint: "Короткий цикл обучения поверх уже существующей модели.",
    longRunning: true,
  },
  fine_tune_saved: {
    title: "Дообучение завершено",
    hint: "Новые веса сохранены.",
  },
  nothing_to_train: {
    title: "Обучение не требуется",
    hint: "Модель уже соответствует текущим данным.",
  },
  saving_to_cloud: {
    title: "Сохранение в хранилище",
    hint: "Копируем артефакты в объектное хранилище (MinIO).",
  },
  job_complete: {
    title: "Готово",
    hint: "Все этапы выполнены. Можно скачать модель.",
  },
  infer_queued: {
    title: "Инференс в очереди",
    hint: "Подготовка к прогону модели на тестовых изображениях.",
  },
  infer_prepare: {
    title: "Распаковка теста",
    hint: "Извлекаем изображения из архива.",
  },
  infer_running: {
    title: "Инференс",
    hint: "Модель обрабатывает изображения.",
    longRunning: true,
  },
  infer_pack: {
    title: "Упаковка результатов",
    hint: "Собираем выходные файлы в архив.",
  },
  infer_upload: {
    title: "Загрузка в хранилище",
    hint: "Сохраняем архив результатов в MinIO.",
  },
  infer_complete: {
    title: "Инференс готов",
    hint: "Можно скачать архив с результатами.",
  },
};

export const TASK_TYPE_UX: Record<string, string> = {
  классификация: "Классификация изображений",
  сегментация: "Сегментация",
};

export const INFER_STEP_ORDER = [
  "infer_queued",
  "infer_prepare",
  "infer_running",
  "infer_pack",
  "infer_upload",
  "infer_complete",
] as const;

export type StepTimelineState = "done" | "current" | "upcoming" | "skipped";

export function stepOrderForKind(kind: "train" | "inference" | undefined): readonly string[] {
  if (kind === "inference") return INFER_STEP_ORDER;
  return JOB_STEP_ORDER;
}

export function getStepTimelineState(
  stepId: string,
  currentStep: string | undefined,
  history: string[] | undefined,
  jobStatus: string
): StepTimelineState {
  const hist = history ?? [];
  const histSet = new Set(hist);

  if (jobStatus === "SUCCESS") {
    return histSet.has(stepId) ? "done" : "skipped";
  }

  if (jobStatus === "FAILURE" || jobStatus === "REVOKED") {
    if (histSet.has(stepId)) return "done";
    return "upcoming";
  }

  const active = currentStep?.trim() || (hist.length ? hist[hist.length - 1] : undefined);
  if (active && stepId === active) return "current";
  if (histSet.has(stepId)) return "done";
  return "upcoming";
}

export function stepProgressPercent(
  step: string | undefined,
  status: string,
  kind?: string
): number {
  if (status === "SUCCESS") return 100;
  if (step === "job_complete" || step === "infer_complete") return 100;
  if (!step) return 5;
  if (kind === "inference") {
    const i = (INFER_STEP_ORDER as readonly string[]).indexOf(step);
    if (i < 0) return 10;
    return Math.min(
      92,
      Math.round(((i + 1) / INFER_STEP_ORDER.length) * 100)
    );
  }
  const i = (JOB_STEP_ORDER as readonly string[]).indexOf(step);
  if (i < 0) return 10;
  return Math.min(92, Math.round(((i + 1) / JOB_STEP_ORDER.length) * 100));
}

export function labelForStep(step: string): { title: string; hint: string; longRunning?: boolean } {
  return (
    JOB_STEP_UX[step] ?? {
      title: "Выполняется",
      hint: step,
    }
  );
}
